from typing import Optional, Dict, List

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from core.data.dataset import SlimPajamaDataset
from core.utils.tokenizer import Tokenizer


class SlimPajamaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: Tokenizer,
        max_length: int = 1024,
        batch_size: int = 32,
        num_workers: int = 4,
        train_size: int = 100000,
        val_size: int = 10000,
        test_size: int = 10000,
        streaming: bool = True,
    ):
        """
        Initialize the SlimPajamaDataModule.

        Args:
            tokenizer (Tokenizer): The tokenizer to use for processing text.
            max_length (int): Maximum sequence length for tokenization.
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of workers for DataLoaders.
            train_size (int): Number of examples in the training set.
            val_size (int): Number of examples in the validation set.
            test_size (int): Number of examples in the test set.
            streaming (bool): Whether to use streaming mode for datasets.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.streaming = streaming

        self.train_dataset: Optional[SlimPajamaDataset] = None
        self.val_dataset: Optional[SlimPajamaDataset] = None
        self.test_dataset: Optional[SlimPajamaDataset] = None

    def setup(self, stage: Optional[str] = None):
        """
        Set up the datasets for each stage (fit, validate, test).

        Args:
            stage (Optional[str]): The stage to set up. Can be 'fit', 'validate', or 'test'.
        """
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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=not self.streaming,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    @staticmethod
    def _collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        max_len = max(ids.size(0) for ids in input_ids)

        input_ids_padded = torch.stack(
            [
                torch.cat(
                    [
                        ids,
                        torch.full(
                            (max_len - ids.size(0),),
                            50256,
                            dtype=ids.dtype,
                            device=ids.device,
                        ),
                    ]
                )
                for ids in input_ids
            ]
        )
        attention_mask_padded = torch.stack(
            [
                torch.cat(
                    [
                        mask,
                        torch.zeros(
                            max_len - mask.size(0), dtype=mask.dtype, device=mask.device
                        ),
                    ]
                )
                for mask in attention_mask
            ]
        )
        labels_padded = torch.stack(
            [
                torch.cat(
                    [
                        label,
                        torch.full(
                            (max_len - label.size(0),),
                            -100,
                            dtype=label.dtype,
                            device=label.device,
                        ),
                    ]
                )
                for label in labels
            ]
        )

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
        }
