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

from core.utils.tokenizer import Tokenizer


def process_batch(args):
    batch, tokenizer, max_length = args
    return [preprocess_example(example, tokenizer, max_length) for example in batch]


def preprocess_example(example: Dict[str, str], tokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    inputs = tokenizer.tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

    return {
        "input_ids": inputs["input_ids"].squeeze(0),
        "attention_mask": inputs["attention_mask"].squeeze(0),
        "labels": inputs["input_ids"].squeeze(0).clone(),
    }


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

        chunk_size = 10000  # Process 10000 examples at a time
        preprocessed_data = []

        for chunk_start in range(0, self.num_examples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, self.num_examples)
            logger.info(f"Processing examples {chunk_start} to {chunk_end}")

            batches = []
            current_batch = []
            for i, example in enumerate(tqdm(self.dataset.skip(chunk_start).take(chunk_end - chunk_start),
                                             desc=f"Collecting data for chunk {chunk_start}-{chunk_end}",
                                             total=chunk_end - chunk_start)):
                current_batch.append(example)
                if len(current_batch) == self.batch_size:
                    batches.append(current_batch)
                    current_batch = []

            if current_batch:
                batches.append(current_batch)

            with mp.Pool(processes=self.num_workers) as pool:
                for result in tqdm(
                        pool.imap(process_batch, [(batch, self.tokenizer, self.max_length) for batch in batches]),
                        total=len(batches), desc=f"Preprocessing chunk {chunk_start}-{chunk_end}"):
                    preprocessed_data.extend(result)

            # Save intermediate results
            intermediate_file = os.path.join(self.cache_dir, f"{self.split}_intermediate_{chunk_start}_{chunk_end}.pkl")
            with open(intermediate_file, 'wb') as f:
                pickle.dump(preprocessed_data, f)
            logger.info(f"Saved intermediate results for examples {chunk_start} to {chunk_end}")

            # Clear preprocessed_data to free up memory
            preprocessed_data = []

        # Combine all intermediate results
        all_data = []
        for chunk_start in range(0, self.num_examples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, self.num_examples)
            intermediate_file = os.path.join(self.cache_dir, f"{self.split}_intermediate_{chunk_start}_{chunk_end}.pkl")
            with open(intermediate_file, 'rb') as f:
                all_data.extend(pickle.load(f))
            os.remove(intermediate_file)  # Remove intermediate file after combining

        end_time = time.time()
        logger.info(f"Dataset loading and preprocessing took {end_time - start_time:.2f} seconds")
        return all_data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)