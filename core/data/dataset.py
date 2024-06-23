import os
import time
from typing import Optional, Iterator, Dict

import torch
from datasets import load_dataset
from loguru import logger
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import GPT2Tokenizer


class SlimPajamaDataset(IterableDataset):
    def __init__(
            self,
            split: str,
            tokenizer: Optional[GPT2Tokenizer] = None,
            max_length: int = 1024,
            num_examples: int = 1000,
            cache_dir: str = "dataset_cache",
            download_timeout: int = 300,
    ):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer or GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.num_examples = num_examples
        self.cache_dir = cache_dir
        self.download_timeout = download_timeout

        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f"{split}_{num_examples}.pkl")

        logger.info(f"Loading {split} dataset...")
        start_time = time.time()
        try:
            self.dataset = load_dataset(
                "cerebras/SlimPajama-627B",
                split=f"{self.split}",
                streaming=True,
                cache_dir="F:\\.cache",
            )
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
        finally:
            end_time = time.time()
            logger.info(f"Dataset loading took {end_time - start_time:.2f} seconds")

        logger.info("SlimPajamaDataset initialization complete.")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterates over the dataset, yielding preprocessed examples one at a time.

        Yields:
            Iterator[Dict[str, torch.Tensor]]: An iterator of preprocessed examples.
        """
        for i, example in tqdm(enumerate(self.dataset), total=self.num_examples, desc="Streaming and Preprocessing"):
            if i >= self.num_examples:
                break
            yield self.preprocess_example(example)

    def preprocess_example(self, example: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """
        Preprocesses a single example from the dataset.

        Args:
            example (Dict[str, str]): A dictionary containing the raw example data.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the preprocessed tensors.
        """
        inputs = self.tokenizer(
            example["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": inputs["input_ids"].squeeze(0).clone()
        }
