import os
from typing import Optional, Dict

import torch
from datasets import load_dataset
from loguru import logger
from torch.utils.data import IterableDataset

from core.utils.tokenizer import Tokenizer


class SlimPajamaDataset(IterableDataset):
    def __init__(
            self,
            split: str,
            tokenizer: Optional[Tokenizer] = None,
            max_length: int = 1024,
            num_examples: int = 1000,
            cache_dir: str = "dataset_cache",
    ):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer or Tokenizer.from_pretrained("gpt2")
        self.max_length = max_length
        self.num_examples = num_examples
        self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)

        logger.info(f"Initializing streaming dataset for {split} split...")
        self.dataset = load_dataset(
            "cerebras/SlimPajama-627B",
            split=f"{self.split}",
            streaming=True,
            cache_dir=self.cache_dir,
        )

    def __iter__(self):
        for i, example in enumerate(self.dataset):
            if i >= self.num_examples:
                break
            yield self.process_example(example['text'])

    def process_example(self, text: str) -> Dict[str, torch.Tensor]:
        tokenized = self.tokenizer.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': tokenized['input_ids'].squeeze(0).clone()
        }

    def __len__(self):
        return self.num_examples
