import math
import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import GPT2Tokenizer

class SlimPajamaDataset(IterableDataset):
    def __init__(self, split: str, subset_size: float = 0.1, max_length: int = 1024, tokenizer: GPT2Tokenizer = None):
        self.split = split
        self.subset_size = subset_size
        self.max_length = max_length

        self.tokenizer = tokenizer if tokenizer else GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.dataset = load_dataset("cerebras/SlimPajama-627B", split=split, streaming=True)

        self.total_examples = 1000000  # Adjust this value based on the estimated total examples in the dataset
        self.subset_examples = math.ceil(self.total_examples * self.subset_size)

    def __len__(self):
        return self.subset_examples

    def __iter__(self):
        for idx, example in enumerate(self.dataset):
            if idx >= self.subset_examples:
                break

            encoding = self.tokenizer(
                example["text"],
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors="pt"
            )
            input_ids = encoding['input_ids'].squeeze()
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            yield input_ids, labels

    @staticmethod
    def collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        return input_ids, labels