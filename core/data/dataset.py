import os
import numpy as np
from typing import Optional, Dict
from datasets import load_dataset
from torch.utils.data import IterableDataset
from loguru import logger
import time
from tqdm import tqdm
import torch
import threading
from queue import Queue
import mmap
from concurrent.futures import ThreadPoolExecutor

from core.utils.tokenizer import Tokenizer


class SlimPajamaDataset(IterableDataset):
    def __init__(
            self,
            split: str,
            tokenizer: Optional[Tokenizer] = None,
            max_length: int = 1024,
            num_examples: int = 1000,
            cache_dir: str = "dataset_cache",
            streaming: bool = False,
            num_workers: int = 4,
            chunk_size: int = 1000,
    ):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer or Tokenizer.from_pretrained("gpt2")
        self.max_length = max_length
        self.num_examples = num_examples
        self.cache_dir = cache_dir
        self.streaming = streaming
        self.num_workers = num_workers
        self.chunk_size = chunk_size

        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir,
                                       f"{split}_{num_examples}{'_streaming' if streaming else ''}.mmap")

        if os.path.exists(self.cache_file):
            logger.info(f"Loading cached {split} dataset...")
            self.load_mmap_data()
        else:
            logger.info(f"Initializing {'streaming' if streaming else 'non-streaming'} dataset for {split} split...")
            self.dataset = load_dataset(
                "cerebras/SlimPajama-627B",
                split=f"{self.split}",
                streaming=True,
                cache_dir="F:\\.cache",
            )
            self.load_and_preprocess_data()

        logger.info(f"Dataset initialization complete for {split} split.")

    def load_mmap_data(self):
        with open(self.cache_file, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), 0)
            header = np.frombuffer(mm[:24], dtype=np.int64)
            self.num_examples, self.max_length = header[:2]
            data_start = 24
            self.data = {
                'input_ids': np.frombuffer(mm[data_start:data_start + self.num_examples * self.max_length * 8],
                                           dtype=np.int64).reshape(self.num_examples, self.max_length),
                'attention_mask': np.frombuffer(mm[data_start + self.num_examples * self.max_length * 8:],
                                                dtype=np.int64).reshape(self.num_examples, self.max_length)
            }
        logger.info(f"Loaded {self.num_examples} examples from cache.")

    def load_and_preprocess_data(self):
        start_time = time.time()

        input_queue = Queue(maxsize=self.num_workers * 2)
        output_queue = Queue(maxsize=self.num_workers * 2)

        def producer():
            for i, example in enumerate(self.dataset):
                if i >= self.num_examples:
                    break
                input_queue.put(example['text'])
            for _ in range(self.num_workers):
                input_queue.put(None)  # Signal to stop

        def consumer():
            while True:
                text = input_queue.get()
                if text is None:
                    break
                preprocessed = self.preprocess_example(text)
                output_queue.put(preprocessed)

        with ThreadPoolExecutor(max_workers=self.num_workers + 1) as executor:
            executor.submit(producer)
            for _ in range(self.num_workers):
                executor.submit(consumer)

            with open(self.cache_file, 'wb') as f:
                # Write header
                np.array([self.num_examples, self.max_length, 0], dtype=np.int64).tofile(f)

                pbar = tqdm(total=self.num_examples, desc=f"Processing {self.split} data")
                for _ in range(0, self.num_examples, self.chunk_size):
                    chunk_data = {'input_ids': [], 'attention_mask': []}
                    for _ in range(min(self.chunk_size, self.num_examples - len(chunk_data['input_ids']))):
                        item = output_queue.get()
                        chunk_data['input_ids'].append(item['input_ids'].numpy())
                        chunk_data['attention_mask'].append(item['attention_mask'].numpy())
                        pbar.update(1)

                    np.array(chunk_data['input_ids'], dtype=np.int64).tofile(f)
                    np.array(chunk_data['attention_mask'], dtype=np.int64).tofile(f)
                pbar.close()

        self.load_mmap_data()  # Load the data we just wrote

        end_time = time.time()
        logger.info(f"Dataset loading and preprocessing took {end_time - start_time:.2f} seconds")

    def preprocess_example(self, text: str) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }

    def __iter__(self):
        for i in range(self.num_examples):
            yield {
                'input_ids': torch.from_numpy(self.data['input_ids'][i]),
                'attention_mask': torch.from_numpy(self.data['attention_mask'][i]),
                'labels': torch.from_numpy(self.data['input_ids'][i]).clone(),
            }

    def __len__(self):
        return self.num_examples