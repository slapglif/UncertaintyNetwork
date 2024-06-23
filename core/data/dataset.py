# core/data/dataset.py

import os
import pickle
from typing import Optional, Dict, List, Union
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset
from transformers import GPT2Tokenizer
from loguru import logger
import time
from tqdm import tqdm
import torch

from core.utils.tokenizer import Tokenizer


class SlimPajamaDataset(Dataset):
    def __init__(
            self,
            split: str,
            tokenizer: Optional[Tokenizer | GPT2Tokenizer] = None,
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

        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f"{split}_{num_examples}.pkl")

        if not streaming:
            if os.path.exists(self.cache_file):
                logger.info(f"Loading cached {split} dataset...")
                with open(self.cache_file, 'rb') as f:
                    self.data = pickle.load(f)
            else:
                logger.info(f"Loading and preprocessing {split} dataset...")
                self.data = self.load_and_preprocess_data()
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.data, f)
            logger.info(f"SlimPajamaDataset initialization complete. Total examples: {len(self.data)}")
        else:
            logger.info(f"Initializing streaming dataset for {split} split...")
            self.streamed_dataset = load_dataset(
                "cerebras/SlimPajama-627B",
                split=f"{self.split}",
                streaming=True,
                cache_dir="F:\\.cache",
            )
            logger.info("Streaming dataset initialized.")

    def load_and_preprocess_data(self) -> List[Dict[str, torch.Tensor]]:
        start_time = time.time()
        try:
            dataset = load_dataset(
                "cerebras/SlimPajama-627B",
                split=f"{self.split}",
                cache_dir="F:\\.cache",
            )
            dataset = dataset.select(range(min(self.num_examples, len(dataset))))

            return [
                self.preprocess_example(example)
                for example in tqdm(
                    dataset,
                    desc=f"Preprocessing {self.split} data",
                    total=len(dataset),
                )
            ]
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
        finally:
            end_time = time.time()
            logger.info(f"Dataset loading and preprocessing took {end_time - start_time:.2f} seconds")

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
        return self.num_examples if self.streaming else len(self.data)

    def __iter__(self):
        if self.streaming:
            return self
        else:
            raise NotImplementedError("Iteration is only supported in streaming mode.")

    def __next__(self):
        if self.streaming:
            example = next(iter(self.streamed_dataset))
            return self.preprocess_example(example)
        else:
            raise NotImplementedError("Next is only supported in streaming mode.")

    def __getitem__(self, idx):
        if self.streaming:
            raise NotImplementedError("Random access is not supported in streaming mode.")
        return self.data[idx]