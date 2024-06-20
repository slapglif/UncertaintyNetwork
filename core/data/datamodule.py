from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from core.data.dataset import SlimPajamaDataset


class SlimPajamaDataModule(LightningDataModule):
    def __init__(self, batch_size: int, subset_size: float, max_length: int):
        super().__init__()
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.batch_size = batch_size
        self.subset_size = subset_size
        self.max_length = max_length

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=SlimPajamaDataset.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=SlimPajamaDataset.collate_fn,
            pin_memory=True,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = SlimPajamaDataset(
                split="train",
                subset_size=self.subset_size,
                max_length=self.max_length,
            )
            self.val_dataset = SlimPajamaDataset(
                split="validation",
                subset_size=self.subset_size,
                max_length=self.max_length,
            )

        if stage == "test" or stage is None:
            self.test_dataset = SlimPajamaDataset(
                split="test",
                subset_size=self.subset_size,
                max_length=self.max_length,
            )
