# data_utils.py
import os
from pathlib import Path
from typing import Optional

import mlflow
import pytorch_lightning as pl
import torch
from datasets import Dataset, load_dataset
from loguru import logger
from pyarrow import Table
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer

from benchmark.engine.augmentation import synonym_replacement, random_insertion, random_swap, random_deletion, \
    apply_augmentations, token_augmentation
from benchmark.engine.config import BenchmarkConfig
from core.data.dataset import SlimPajamaDataset
from core.utils.tokenizer import Tokenizer


class CachedDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length, cache_dir, arrow_table: Table):
        super().__init__(arrow_table)
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        cache_file = self.cache_dir / f"item_{idx}.pt"
        if cache_file.exists():
            item = torch.load(cache_file)
            mlflow.log_artifact(str(cache_file), "dataset_cache")
            return item

        item = self.dataset[idx]
        encoded = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        result = {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": encoded["input_ids"].squeeze(),
        }
        torch.save(result, cache_file)
        mlflow.log_artifact(str(cache_file), "dataset_cache")
        return result


class FullTrainingDataModule(pl.LightningDataModule):
    def __init__(self, config, dataset_name: str, cache_dir: str):
        super().__init__()
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.config = config
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

    def setup(self, stage: str = None):
        if self.dataset_name == "qa":
            dataset = load_dataset("squad")
        elif self.dataset_name == "coding":
            dataset = load_dataset("codeparrot/github-code")
        elif self.dataset_name == "generalization":
            dataset = load_dataset("winogrande", "winogrande_xl")
        elif self.dataset_name == "medical":
            dataset = load_dataset("medical_questions_pairs")
        else:  # misc
            dataset = load_dataset("wikitext", "wikitext-2-v1")

        self.train_dataset = CachedDataset(
            dataset["train"], self.tokenizer, self.config.max_length,
            os.path.join(self.cache_dir, self.dataset_name, "train")
        )
        self.val_dataset = CachedDataset(
            dataset["validation"], self.tokenizer, self.config.max_length,
            os.path.join(self.cache_dir, self.dataset_name, "validation")
        )
        self.test_dataset = CachedDataset(
            dataset["test"], self.tokenizer, self.config.max_length,
            os.path.join(self.cache_dir, self.dataset_name, "test")
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)


class Wikitext2Dataset(Dataset):
    def __init__(self, tokenizer, data, max_length, arrow_table: Table = None):
        super().__init__(arrow_table)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self.preprocess_data(data)

    def preprocess_data(self, data):
        examples = []
        for item in data:
            if item["text"].strip():  # Skip empty texts
                encodings = self.tokenizer(
                    item["text"],
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                )
                examples.append(
                    {
                        "input_ids": torch.tensor(encodings["input_ids"]),
                        "attention_mask": torch.tensor(encodings["attention_mask"]),
                    }
                )
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        labels = item["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": labels,
        }


class Wikitext2DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=32, max_length=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def setup(self, stage=None):
        dataset = load_dataset("wikitext", "wikitext-2-v1")

        self.train_dataset = Wikitext2Dataset(
            self.tokenizer, dataset["train"], self.max_length
        )
        self.val_dataset = Wikitext2Dataset(
            self.tokenizer, dataset["validation"], self.max_length
        )
        self.test_dataset = Wikitext2Dataset(
            self.tokenizer, dataset["test"], self.max_length
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True,
        )


class BenchmarkDataModule(pl.LightningDataModule):
    def __init__(self, config: BenchmarkConfig):
        super().__init__()
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.config = config
        self.tokenizer = Tokenizer(config.tokenizer_path)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = SlimPajamaDataset(
                split="train",
                tokenizer=self.tokenizer,
                max_length=self.config.max_length,
                num_examples=self.config.batch_size * 1000  # Adjust as needed
            )
            self.val_dataset = SlimPajamaDataset(
                split="validation",
                tokenizer=self.tokenizer,
                max_length=self.config.max_length,
                num_examples=self.config.batch_size * 100  # Adjust as needed
            )

        if stage == 'test' or stage is None:
            self.test_dataset = SlimPajamaDataset(
                split="test",
                tokenizer=self.tokenizer,
                max_length=self.config.max_length,
                num_examples=self.config.batch_size * 100  # Adjust as needed
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True
        )


class UncertaintyAwareDataset(Dataset):
    def __init__(self, base_dataset: SlimPajamaDataset, uncertainty_processor: callable, arrow_table: Table):
        super().__init__(arrow_table)
        self.base_dataset = base_dataset
        self.uncertainty_processor = uncertainty_processor

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        uncertainty_data = self.uncertainty_processor(item)
        item.update(uncertainty_data)
        return item


class AugmentedDataset(Dataset):
    def __init__(self, base_dataset: SlimPajamaDataset, config: BenchmarkConfig, tokenizer: PreTrainedTokenizer,
                 arrow_table: Table):
        super().__init__(arrow_table)
        self.base_dataset = base_dataset
        self.config = config
        self.tokenizer = tokenizer
        self.augmentation_pipeline = [
            lambda x: synonym_replacement(x, n=2),
            lambda x: random_insertion(x, n=1),
            lambda x: random_swap(x, n=1),
            lambda x: random_deletion(x, p=0.1)
        ]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]

        # Text augmentation
        original_text = self.tokenizer.decode(item['input_ids'], skip_special_tokens=True)
        augmented_texts = apply_augmentations(original_text, self.augmentation_pipeline, num_augmentations=1)
        augmented_inputs = self.tokenizer(augmented_texts[0], truncation=True, padding='max_length',
                                          max_length=self.config.max_length)

        # Token augmentation
        augmented_input_ids = token_augmentation(torch.tensor(augmented_inputs['input_ids']), self.tokenizer)

        return {
            'input_ids': augmented_input_ids,
            'attention_mask': torch.tensor(augmented_inputs['attention_mask']),
            'labels': item['labels']
        }


def analyze_data_distribution(dataset: SlimPajamaDataset):
    logger.info("Analyzing data distribution...")
    token_freqs = {}
    seq_lengths = []

    for item in tqdm(dataset, desc="Analyzing data"):
        seq_lengths.append(len(item['input_ids']))
        for token in item['input_ids']:
            token_freqs[token] = token_freqs.get(token, 0) + 1

    logger.info(f"Vocabulary size: {len(token_freqs)}")
    logger.info(f"Average sequence length: {sum(seq_lengths) / len(seq_lengths):.2f}")
    logger.info(f"Max sequence length: {max(seq_lengths)}")
    logger.info(f"Min sequence length: {min(seq_lengths)}")

    return token_freqs, seq_lengths


def get_dataset_loader(dataset_name: str, config, cache_dir: str) -> BenchmarkDataModule:
    """Get the appropriate dataset loader based on the dataset name."""
    return BenchmarkDataModule(config)
