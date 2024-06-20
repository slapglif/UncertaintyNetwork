from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

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
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Set the pad_token to the eos_token
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = SlimPajamaDataset(
                split="train",
                subset_size=self.subset_size,
                max_length=self.max_length,
                tokenizer=self.tokenizer,
            )
            self.val_dataset = SlimPajamaDataset(
                split="validation",
                subset_size=self.subset_size,
                max_length=self.max_length,
                tokenizer=self.tokenizer,
            )

        if stage == "test" or stage is None:
            self.test_dataset = SlimPajamaDataset(
                split="test",
                subset_size=self.subset_size,
                max_length=self.max_length,
                tokenizer=self.tokenizer,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=4,
            pin_memory=True,
        )

    def collate_fn(self, batch):
        input_ids = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        input_ids = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding="longest",
            return_tensors="pt"
        )["input_ids"]

        labels = self.tokenizer.pad(
            {"input_ids": labels},
            padding="longest",
            return_tensors="pt"
        )["input_ids"]

        return input_ids, labels