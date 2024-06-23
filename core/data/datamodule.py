from typing import Optional, Dict, Any
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
            streaming: bool = False,
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
        """
        Create and return the training DataLoader.

        Returns:
            DataLoader: The training DataLoader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=not self.streaming,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create and return the validation DataLoader.

        Returns:
            DataLoader: The validation DataLoader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create and return the test DataLoader.

        Returns:
            DataLoader: The test DataLoader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch: list) -> Dict[str, Any]:
        """
        Custom collate function to process batches.

        Args:
            batch (list): A list of samples from the dataset.

        Returns:
            Dict[str, Any]: A dictionary containing the processed batch.
        """
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]

        # Pad sequences to the maximum length in the batch
        max_len = max(len(ids) for ids in input_ids)
        input_ids = [ids + [self.tokenizer.pad_token_id] * (max_len - len(ids)) for ids in input_ids]
        attention_mask = [mask + [0] * (max_len - len(mask)) for mask in attention_mask]

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(input_ids).clone(),  # Use input_ids as labels for language modeling
        }