import itertools
from typing import Optional, Dict

import torch
from datasets import load_dataset
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
            streaming: bool = False,
    ):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer or Tokenizer.from_pretrained("gpt2")
        self.max_length = max_length
        self.num_examples = num_examples
        self.cache_dir = cache_dir
        self.streaming = streaming

        logger.info(f"Initializing {'streaming' if streaming else 'non-streaming'} dataset for {split} split...")
        self.dataset = load_dataset(
            "cerebras/SlimPajama-627B",
            split=f"{self.split}",
            streaming=self.streaming,
            cache_dir="F:\\.cache",
        )
        if not self.streaming:
            self.dataset = self.dataset.select(range(min(self.num_examples, len(self.dataset))))
        logger.info(f"Dataset initialized for {split} split.")

    def __iter__(self):
        if self.streaming:
            return self.infinite_iterator()
        else:
            return map(self.preprocess_example, self.dataset)

    def infinite_iterator(self):
        iterator = map(self.preprocess_example, self.dataset)
        return itertools.cycle(iterator)

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

    def __len__(self):
        return self.num_examples if self.streaming else len(self.dataset)
