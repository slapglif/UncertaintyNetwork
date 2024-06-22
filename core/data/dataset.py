import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from typing import Optional

class SlimPajamaDataset(Dataset):
    def __init__(
        self,
        split: str,
        tokenizer: Optional[GPT2Tokenizer] = None,
        max_length: int = 1024,
        num_examples: int = 10000,
    ):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer or GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.num_examples = num_examples

        self.dataset = load_dataset("cerebras/SlimPajama-627B", split=self.split, streaming=True, cache_dir="F:\\.cache")
        self.dataset = self.dataset.take(self.num_examples)
        self.dataset = self.dataset.map(self.preprocess_example, batched=True, remove_columns=["text"])

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return self.dataset[idx]

    def preprocess_example(self, examples):
        input_ids = self.tokenizer(examples["text"], truncation=True, max_length=self.max_length, padding="max_length")["input_ids"]
        attention_mask = [[1] * len(ids) + [0] * (self.max_length - len(ids)) for ids in input_ids]
        labels = [ids.copy() for ids in input_ids]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

