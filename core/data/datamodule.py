# core/data/datamodule.py

from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from core.data.dataset import SlimPajamaDataset
from core.utils.tokenizer import Tokenizer


class SlimPajamaDataModule(LightningDataModule):
    def __init__(
            self,
            tokenizer: Tokenizer,
            max_length: int = 1024,
            batch_size: int = 32,
            num_workers: int = 4,
            train_size: int = 100000,
            val_size: int = 10000,
            test_size: int = 10000,
            streaming: bool = False,
    ):
        super().__init__()
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.streaming = streaming

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = SlimPajamaDataset(
                split="train",
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                num_examples=self.train_size,
            )
            self.val_dataset = SlimPajamaDataset(
                split="validation",
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                num_examples=self.val_size,
            )

        if stage == "test" or stage is None:
            self.test_dataset = SlimPajamaDataset(
                split="test",
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                num_examples=self.test_size,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=not self.streaming,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
