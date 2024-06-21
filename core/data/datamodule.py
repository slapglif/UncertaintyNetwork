from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset


class SlimPajamaDataModule(LightningDataModule):
    def __init__(self, batch_size: int, subset_size: float, max_length: int):
        super().__init__()
        self.batch_size = batch_size
        self.subset_size = subset_size
        self.max_length = max_length
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset once during initialization
        self.dataset = load_dataset(
            "cerebras/SlimPajama-627B",
            streaming=True,
            split="train"
        )

    def setup(self, stage=None):
        pass  # We don't need to do anything here as we've loaded the dataset in __init__

    def train_dataloader(self):
        return DataLoader(
            self.dataset.shuffle(seed=42).take(int(self.subset_size * 1_000_000)),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset.shuffle(seed=42).skip(int(self.subset_size * 1_000_000)).take(10_000),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=4,
        )

    def collate_fn(self, examples):
        texts = [ex['text'] for ex in examples]
        encoding = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoding['input_ids']
        labels = input_ids.clone()

        # Mask out the padding tokens for loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100

        return input_ids, labels