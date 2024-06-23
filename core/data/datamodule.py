from typing import Optional, List, Dict

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from core.data.dataset import SlimPajamaDataset


class SlimPajamaDataModule(LightningDataModule):
    def __init__(
            self,
            tokenizer: GPT2Tokenizer,
            max_length: int = 1024,
            train_num_examples: int = 100000,
            val_num_examples: int = 10000,
            test_num_examples: int = 10000,
            batch_size: int = 32,
            streaming: bool = True
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_num_examples = train_num_examples
        self.val_num_examples = val_num_examples
        self.test_num_examples = test_num_examples
        self.batch_size = batch_size
        self.streaming = streaming

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
        """
        Creates and returns the training DataLoader.

        Returns:
            DataLoader: The DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=11 if self.streaming else 4,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self) -> DataLoader:
        """
        Creates and returns the validation DataLoader.

        Returns:
            DataLoader: The DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=11 if self.streaming else 4,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=True
        )

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for batching the data.

        Args:
            batch (List[Dict[str, torch.Tensor]]): A list of dictionaries containing the data samples.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the batched tensors.
        """
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }