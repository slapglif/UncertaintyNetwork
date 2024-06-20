import math

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import GPT2Tokenizer


class SlimPajamaDataset(IterableDataset):
    def __init__(self, split: str, subset_size: float = 0.1, max_length: int = 1024):
        self.split = split
        self.subset_size = subset_size
        self.max_length = max_length

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.dataset = load_dataset("cerebras/SlimPajama-627B", split=split, cache_dir="F:\\.cache", streaming=True)

        self.total_examples = 1000000  # Adjust this value based on the estimated total examples in the dataset
        self.subset_examples = math.ceil(self.total_examples * self.subset_size)

    def __len__(self):
        return self.total_examples

    def __iter__(self):
        for idx, example in enumerate(self.dataset):
            if idx >= self.subset_examples:
                break

            tokenized_text = self.tokenizer(example["text"])["input_ids"]
            for i in range(0, len(tokenized_text), self.max_length):
                chunk = tokenized_text[i:i + self.max_length]
                if len(chunk) < 2:
                    continue
                x = torch.tensor(chunk[:-1])
                y = torch.tensor(chunk[1:])
                yield x, y

    @staticmethod
    def collate_fn(batch):
        data = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
        targets = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0)
        return data, targets
