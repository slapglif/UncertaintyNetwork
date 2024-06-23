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
from contextlib import contextmanager

from core.utils.tokenizer import Tokenizer


@contextmanager
def limit_num_open_files(max_open):
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max_open, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


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

        batches = []
        current_batch = []
        for i, example in enumerate(tqdm(self.dataset, desc=f"Collecting {self.split} data", total=self.num_examples)):
            if i >= self.num_examples:
                break
            current_batch.append(example)
            if len(current_batch) == self.batch_size:
                batches.append(current_batch)
                current_batch = []

        if current_batch:
            batches.append(current_batch)

        preprocessed_data = []
        chunk_size = min(10, len(batches))  # Process 10 batches at a time, or fewer if there are less than 10 batches

        with limit_num_open_files(1024):  # Limit the number of open files
            for i in range(0, len(batches), chunk_size):
                chunk = batches[i:i + chunk_size]
                with mp.Pool(processes=self.num_workers) as pool:
                    for result in tqdm(
                            pool.imap(process_batch, [(batch, self.tokenizer, self.max_length) for batch in chunk]),
                            total=len(chunk),
                            desc=f"Preprocessing chunk {i // chunk_size + 1}/{len(batches) // chunk_size + 1}"):
                        preprocessed_data.extend(result)

        end_time = time.time()
        logger.info(f"Dataset loading and preprocessing took {end_time - start_time:.2f} seconds")
        return preprocessed_data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)