import math
from typing import Optional, Iterator, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import GPT2Tokenizer


class SlimPajamaDataset(IterableDataset):
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

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        dataset = load_dataset("cerebras/SlimPajama-627B", split=self.split, streaming=True, cache_dir="F:\\.cache")

        with tqdm(total=self.num_examples, desc=f"Loading {self.split} data") as pbar:
            for idx, example in enumerate(dataset):
                if idx >= self.num_examples:
                    break

                input_ids = self.tokenizer.encode(example["text"], add_special_tokens=False)
                input_ids = input_ids[:self.max_length]
                attention_mask = [1] * len(input_ids)

                # Pad input_ids and attention_mask to max_length
                input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
                attention_mask = attention_mask + [0] * (self.max_length - len(attention_mask))

                labels = input_ids.copy()
                labels_attention_mask = attention_mask.copy()

                input_ids = torch.tensor(input_ids, dtype=torch.long)
                attention_mask = torch.tensor(attention_mask, dtype=torch.long)
                labels = torch.tensor(labels, dtype=torch.long)
                labels_attention_mask = torch.tensor(labels_attention_mask, dtype=torch.long)

                yield input_ids, attention_mask, labels, labels_attention_mask

                pbar.update(1)