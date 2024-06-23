import os
import pickle
import time
from typing import Optional, Dict, List

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
            streaming: bool = False,
    ):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer or GPT2Tokenizer.from_pretrained("gpt2")
        self.max_length = max_length
        self.num_examples = num_examples
        self.cache_dir = cache_dir
        self.streaming = streaming

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
                streaming=True,  # Always use streaming for initial load
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
        preprocessed_data = []
        try:
            for i, example in enumerate(
                    tqdm(self.dataset, desc=f"Preprocessing {self.split} data", total=self.num_examples)):
                if i >= self.num_examples:
                    break
                preprocessed_data.append(self.preprocess_example(example))

                # Save intermediate results every 1000 examples
                if (i + 1) % 1000 == 0:
                    logger.info(f"Saving intermediate results: {i + 1} examples processed")
                    with open(self.cache_file, 'wb') as f:
                        pickle.dump(preprocessed_data, f)
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
        finally:
            end_time = time.time()
            logger.info(f"Dataset loading and preprocessing took {end_time - start_time:.2f} seconds")
        return preprocessed_data

    def preprocess_example(self, example: Dict[str, str]) -> Dict[str, torch.Tensor]:
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
            "labels": inputs["input_ids"].squeeze(0).clone(),
        }

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)
