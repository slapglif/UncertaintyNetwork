import os
import pickle
from typing import Optional, Dict, List
from datasets import load_dataset
from torch.utils.data import IterableDataset
from loguru import logger
import time
from tqdm import tqdm
import torch
import multiprocessing as mp
from functools import partial

from core.utils.tokenizer import Tokenizer


class SlimPajamaDataset(IterableDataset):
    def __init__(
            self,
            split: str,
            tokenizer: Optional[Tokenizer] = None,
            max_length: int = 1024,
            num_examples: int = 1000,
            cache_dir: str = "dataset_cache",
            streaming: bool = False,
            batch_size: int = 1000,
            num_workers: int = 4
    ):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer or Tokenizer.from_pretrained("gpt2")
        self.max_length = max_length
        self.num_examples = num_examples
        self.cache_dir = cache_dir
        self.streaming = streaming
        self.batch_size = batch_size
        self.num_workers = num_workers

        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f"{split}_{num_examples}{'_streaming' if streaming else ''}.pkl")

        if os.path.exists(self.cache_file):
            logger.info(f"Loading cached {split} dataset...")
            with open(self.cache_file, 'rb') as f:
                self.data = pickle.load(f)
            logger.info(f"Loaded {len(self.data)} examples from cache.")
        else:
            logger.info(f"Initializing {'streaming' if streaming else 'non-streaming'} dataset for {split} split...")
            self.dataset = load_dataset(
                "cerebras/SlimPajama-627B",
                split=f"{self.split}",
                streaming=True,
                cache_dir="F:\\.cache",
            )
            self.data = self.load_and_preprocess_data()
            logger.info(f"Saving {len(self.data)} examples to cache...")
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.data, f)
            logger.info(f"Cached {len(self.data)} examples.")

        logger.info(f"Dataset initialization complete for {split} split.")

    def load_and_preprocess_data(self) -> List[Dict[str, torch.Tensor]]:
        start_time = time.time()

        def process_batch(batch):
            return [self.preprocess_example(_example) for _example in batch]

        with mp.Pool(processes=self.num_workers) as pool:
            batches = []
            current_batch = []
            for i, example in enumerate(
                    tqdm(self.dataset, desc=f"Collecting {self.split} data", total=self.num_examples)):
                if i >= self.num_examples:
                    break
                current_batch.append(example)
                if len(current_batch) == self.batch_size:
                    batches.append(current_batch)
                    current_batch = []

            if current_batch:
                batches.append(current_batch)

            preprocessed_data = []
            for result in tqdm(pool.imap(process_batch, batches), total=len(batches), desc="Preprocessing batches"):
                preprocessed_data.extend(result)

        end_time = time.time()
        logger.info(f"Dataset loading and preprocessing took {end_time - start_time:.2f} seconds")
        return preprocessed_data

    def preprocess_example(self, example: Dict[str, str]) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer.tokenizer(
            example["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": inputs["input_ids"].squeeze(0).clone(),
        }

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)