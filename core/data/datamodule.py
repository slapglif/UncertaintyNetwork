import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from typing import Optional

from core.data.dataset import SlimPajamaDataset


class SlimPajamaDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer: GPT2Tokenizer,
        max_length: int = 1024,
        train_num_examples: int = 100000,
        val_num_examples: int = 10000,
        test_num_examples: int = 10000,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_num_examples = train_num_examples
        self.val_num_examples = val_num_examples
        self.test_num_examples = test_num_examples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = SlimPajamaDataset(
                split="train",
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                num_examples=self.train_num_examples,
            )
            self.val_dataset = SlimPajamaDataset(
                split="validation",
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                num_examples=self.val_num_examples,
            )

        if stage == "test" or stage is None:
            self.test_dataset = SlimPajamaDataset(
                split="test",
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                num_examples=self.test_num_examples,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        input_ids, attention_mask, labels, labels_attention_mask = zip(*batch)
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        labels = torch.stack(labels)
        labels_attention_mask = torch.stack(labels_attention_mask)
        return input_ids, attention_mask, labels, labels_attention_mask