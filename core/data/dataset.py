import os
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Optional

from datasets import load_dataset
from loguru import logger
from torch.utils.data import IterableDataset

from core.utils.tokenizer import Tokenizer


class SlimPajamaDataset(IterableDataset):
    def __init__(
            self,
            split: str,
            tokenizer: Optional[Tokenizer] = None,
            max_length: int = 1024,
            num_examples: int = 1000,
            cache_dir: str = "dataset_cache",
            num_proc: int = 8,
            buffer_size: int = 1000
    ):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer or Tokenizer.from_pretrained("gpt2")
        self.max_length = max_length
        self.num_examples = num_examples
        self.cache_dir = cache_dir
        self.num_proc = num_proc
        self.buffer_size = buffer_size

        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f"{split}_{num_examples}_streaming.npz")

        logger.info(f"Initializing streaming dataset for {split} split...")
        self.dataset = load_dataset(
            "cerebras/SlimPajama-627B",
            split=f"{self.split}",
            streaming=True,
            cache_dir="F:\\.cache",
        )
        self.tokenize_queue = Queue(maxsize=self.buffer_size)
        self.process_queue = Queue(maxsize=self.buffer_size)
        self.stop_event = threading.Event()

        self.executor = ThreadPoolExecutor(max_workers=self.num_proc)
        self.tokenize_future = self.executor.submit(self.tokenize_worker)
        self.process_futures = [self.executor.submit(self.process_worker) for _ in range(self.num_proc - 1)]

    def tokenize_worker(self):
        for i, example in enumerate(self.dataset):
            if i >= self.num_examples:
                break
            self.tokenize_queue.put(example['text'])
        for _ in range(self.num_proc - 1):
            self.tokenize_queue.put(None)

    def process_worker(self):
        while not self.stop_event.is_set():
            text = self.tokenize_queue.get()
            if text is None:
                break
            tokenized = self.tokenizer.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            self.process_queue.put({
                'input_ids': tokenized['input_ids'].squeeze(0),
                'attention_mask': tokenized['attention_mask'].squeeze(0)
            })

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop_event.is_set():
            raise StopIteration
        try:
            item = self.process_queue.get(timeout=5)
            item['labels'] = item['input_ids'].clone()
            return item
        except Exception as e:
            logger.error(e)
            if self.process_queue.empty():
                self.stop_event.set()
                raise StopIteration

    def __len__(self):
        return self.num_examples

    def close(self):
        self.stop_event.set()
        self.executor.shutdown(wait=True)
