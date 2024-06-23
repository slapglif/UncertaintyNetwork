import os
import numpy as np
from typing import Optional, Dict
from datasets import load_dataset
from torch.utils.data import Dataset
from loguru import logger
import time
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from core.utils.tokenizer import Tokenizer

class SlimPajamaDataset(Dataset):
    def __init__(
            self,
            split: str,
            tokenizer: Optional[Tokenizer] = None,
            max_length: int = 1024,
            num_examples: int = 1000,
            cache_dir: str = "dataset_cache",
            num_proc: int = 8
    ):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer or Tokenizer.from_pretrained("gpt2")
        self.max_length = max_length
        self.num_examples = num_examples
        self.cache_dir = cache_dir
        self.num_proc = num_proc

        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f"{split}_{num_examples}.npz")

        if os.path.exists(self.cache_file):
            logger.info(f"Loading cached {split} dataset...")
            self.data = np.load(self.cache_file, allow_pickle=True)
            self.data = {k: torch.from_numpy(v) for k, v in self.data.items()}
            logger.info(f"Loaded {len(self.data['input_ids'])} examples from cache.")
        else:
            logger.info(f"Initializing dataset for {split} split...")
            self.load_and_preprocess_data()

        logger.info(f"Dataset initialization complete for {split} split.")

    def load_and_preprocess_data(self):
        start_time = time.time()

        dataset = load_dataset(
            "cerebras/SlimPajama-627B",
            split=f"{self.split}",
            cache_dir="F:\\.cache",
        )

        # Use dataset's map function for efficient parallel processing
        tokenized_dataset = dataset.select(range(self.num_examples)).map(
            self.tokenize_function,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
        )

        # Convert to pytorch tensors
        self.data = {
            'input_ids': torch.tensor(tokenized_dataset['input_ids']),
            'attention_mask': torch.tensor(tokenized_dataset['attention_mask']),
        }

        # Save to cache
        np.savez_compressed(self.cache_file, **{k: v.numpy() for k, v in self.data.items()})

        end_time = time.time()
        logger.info(f"Dataset loading and preprocessing took {end_time - start_time:.2f} seconds")

    def tokenize_function(self, examples):
        return self.tokenizer.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

    def __getitem__(self, idx):
        return {
            'input_ids': self.data['input_ids'][idx],
            'attention_mask': self.data['attention_mask'][idx],
            'labels': self.data['input_ids'][idx].clone(),
        }

    def __len__(self):
        return len(self.data['input_ids'])