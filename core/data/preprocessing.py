# core/data/preprocessing.py
import os
import random
from typing import Dict, Union, Optional, Generator, List

import nlpaug.augmenter.char as char_augmenter
import nlpaug.augmenter.sentence as sentence_augmenter
import nlpaug.augmenter.word as word_augmenter
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from loguru import logger
from nlpaug.augmenter.word import BackTranslationAug
from nltk.corpus import stopwords
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

# Download the stopwords corpus
stop_words = set(stopwords.words("english"))  # Use a set for faster lookup

# Import the datasets logging module
import datasets.utils.logging as ds_logging


class DataProcessor:
    """
    Handles text preprocessing and data augmentation using popular libraries, with batch caching.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for text processing.
        data_path (str, optional): The path to the dataset file. If provided, the dataset will be loaded from this path.
        dataset_name (str, optional): The name of the dataset to load from the Hugging Face Datasets library.
        dataset_config (str, optional): The configuration of the dataset to load from the Hugging Face Datasets library.
        augmentation_config (Dict[str, Union[float, int]], optional): A dictionary containing the configuration for data augmentation.
            The keys should include:
                - 'char_swap_ratio', 'char_delete_ratio', 'char_insert_ratio', 'char_swap_probability'
                - 'word_swap_ratio', 'word_delete_ratio', 'word_insert_ratio', 'word_swap_probability'
                - 'sentence_swap_ratio', 'sentence_delete_ratio', 'sentence_insert_ratio', 'sentence_swap_probability'
                - 'synonym_augmentation_probability'
                - 'backtranslation_augmentation_probability', 'backtranslation_language'
        cache_dir (str, optional): The directory to use for caching processed batches. Defaults to "processed_batches".
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            data_path: Optional[str] = None,
            dataset_name: Optional[str] = None,
            dataset_config: Optional[str] = None,
            augmentation_config: Optional[Dict[str, Union[float, int]]] = None,
            cache_dir: str = "processed_batches",
    ):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.augmentation_config = augmentation_config or {}
        self.cache_dir = cache_dir

        # Set up the augmentation pipelines
        self.char_augmenter = self._setup_char_augmenter()
        self.word_augmenter = self._setup_word_augmenter()
        self.sentence_augmenter = self._setup_sentence_augmenter()
        self.synonym_augmenter = self._setup_synonym_augmenter()

        # Create the cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_dataset(self) -> Union[Dataset, DatasetDict]:
        """
        Load the dataset from the specified path or using the Hugging Face Datasets library.

        Returns:
            Union[Dataset, DatasetDict]: The loaded dataset.

        Raises:
            ValueError: If neither `data_path` nor `dataset_name` and `dataset_config` are provided.
        """
        if self.data_path:
            logger.info(f"Loading dataset from path: {self.data_path}")
            return Dataset.from_json(self.data_path)
        elif self.dataset_name and self.dataset_config:
            logger.info(f"Loading dataset from Hugging Face: {self.dataset_name}, {self.dataset_config}")
            return load_dataset(self.dataset_name, self.dataset_config)
        else:
            raise ValueError("Either 'data_path' or 'dataset_name' and 'dataset_config' must be provided.")

    def _setup_char_augmenter(self) -> List[char_augmenter.CharAugmenter]:
        """
        Set up the character-level augmentation pipeline using nlpaug.

        Returns:
            List[char_augmenter.CharAugmenter]: The list of character-level augmenters.
        """
        char_swap_ratio = self.augmentation_config.get("char_swap_ratio", 0.1)
        char_delete_ratio = self.augmentation_config.get("char_delete_ratio", 0.1)
        char_insert_ratio = self.augmentation_config.get("char_insert_ratio", 0.1)
        char_swap_probability = self.augmentation_config.get("char_swap_probability", 0.5)

        char_swap_augmenter = char_augmenter.RandomCharAug(
            action="swap", name="RandomCharSwap", aug_char_min=1, aug_char_max=2
        )
        char_delete_augmenter = char_augmenter.RandomCharAug(
            action="delete", name="RandomCharDelete", aug_char_min=1, aug_char_max=2
        )
        char_insert_augmenter = char_augmenter.RandomCharAug(
            action="insert", name="RandomCharInsert", aug_char_min=1, aug_char_max=2,
        )

        return [char_swap_augmenter, char_delete_augmenter, char_insert_augmenter]

    def _setup_word_augmenter(self) -> List[word_augmenter.WordAugmenter]:
        """
        Set up the word-level augmentation pipeline using nlpaug.

        Returns:
            List[word_augmenter.WordAugmenter]: The list of word-level augmenters.
        """
        word_swap_ratio = self.augmentation_config.get("word_swap_ratio", 0.1)
        word_delete_ratio = self.augmentation_config.get("word_delete_ratio", 0.1)
        # word_insert_ratio = self.augmentation_config.get("word_insert_ratio", 0.1)  # Remove this line
        word_swap_probability = self.augmentation_config.get("word_swap_probability", 0.5)

        word_swap_augmenter = word_augmenter.RandomWordAug(
            action="swap", aug_p=word_swap_ratio, aug_min=1, aug_max=2, tokenizer=self.tokenizer
        )
        word_delete_augmenter = word_augmenter.RandomWordAug(
            action="delete", aug_p=word_delete_ratio, aug_min=1, aug_max=2
        )
        # word_insert_augmenter = word_augmenter.RandomWordAug(  # Remove this block
        #     action="insert", aug_p=word_insert_ratio, aug_min=1, aug_max=2
        # )

        return [word_swap_augmenter, word_delete_augmenter]  # Remove word_insert_augmenter from the list

    def _setup_sentence_augmenter(self) -> List[sentence_augmenter.SentenceAugmenter]:
        """
        Set up the sentence-level augmentation pipeline using nlpaug.

        Returns:
            List[sentence_augmenter.SentenceAugmenter]: The list of sentence-level augmenters.
        """
        sentence_swap_ratio = self.augmentation_config.get("sentence_swap_ratio", 0.1)
        sentence_delete_ratio = self.augmentation_config.get("sentence_delete_ratio", 0.1)
        sentence_insert_ratio = self.augmentation_config.get("sentence_insert_ratio", 0.1)
        sentence_swap_probability = self.augmentation_config.get("sentence_swap_probability", 0.5)

        sentence_swap_augmenter = sentence_augmenter.RandomSentAug(
            aug_p=sentence_swap_ratio, aug_min=1, aug_max=2
        )
        sentence_delete_augmenter = sentence_augmenter.RandomSentAug(
            aug_p=sentence_delete_ratio, aug_min=1, aug_max=2
        )
        sentence_insert_augmenter = sentence_augmenter.RandomSentAug(
            aug_p=sentence_insert_ratio, aug_min=1, aug_max=2
        )

        return [
            sentence_swap_augmenter,
            sentence_delete_augmenter,
            sentence_insert_augmenter,
        ]

    def _setup_synonym_augmenter(self) -> word_augmenter.SynonymAug:
        """
        Set up the synonym augmentation pipeline using nlpaug.

        Returns:
            word_augmenter.SynonymAug: The synonym augmenter.
        """
        synonym_augmentation_probability = self.augmentation_config.get("synonym_augmentation_probability", 0.1)

        return word_augmenter.SynonymAug(aug_p=synonym_augmentation_probability)

    def _setup_backtranslation_augmenter(self) -> BackTranslationAug:
        """
        Set up the backtranslation augmentation pipeline using nlpaug.

        Returns:
            naw.BackTranslationAug: The backtranslation augmenter.
        """
        backtranslation_augmentation_probability = self.augmentation_config.get(
            "backtranslation_augmentation_probability", 0.1)
        backtranslation_language = self.augmentation_config.get("backtranslation_language", "fr")

        return BackTranslationAug(
            from_model_name="facebook/bart-large-cnn",
            to_model_name="facebook/bart-large-cnn",
        )

    def load_and_process_dataset(self) -> Union[Dataset, DatasetDict]:
        """
        Loads the dataset and applies preprocessing using generators, vectorized operations, and batch caching.

        Returns:
            Union[Dataset, DatasetDict]: The loaded and preprocessed dataset.
        """
        dataset = self._load_dataset()

        processed_dataset = DatasetDict()
        # Create a single tqdm bar for the entire data processing
        with tqdm(total=len(dataset), desc="Processing Dataset", leave=False) as pbar_splits:
            for split in dataset.keys():
                logger.info(f"✨ Preprocessing {split} split... ✨")
                processed_data = {
                    "text": list(self._preprocess_split(dataset[split], split))}  # Pass pbar to _preprocess_split
                processed_dataset[split] = Dataset.from_dict(processed_data)
                pbar_splits.update(1)  # Update the progress bar after processing each split

        logger.info("✅ Dataset preprocessing complete! 🎉")
        return processed_dataset

    def _preprocess_split(self, dataset_split: Dataset, split: str) -> Generator[str, None, None]:
        """
        Preprocesses a single split of the dataset using generators, vectorized operations, and batch caching.

        Args:
            dataset_split (Dataset): A single split of the dataset.
            split (str): The name of the split (e.g., "train", "test").

        Returns:
            Generator[str, None, None]: A generator yielding preprocessed text data.
        """
        logger.info(f"✨ Preprocessing {split} split... ✨")  # Log split preprocessing start

        # Apply filter only once at the beginning
        dataset_split = dataset_split.filter(lambda example: len(example["text"]) > 0)
        logger.info(f"Filtered empty examples from {split} split ✅")

        batch_size = 1000  # Adjust batch size as needed

        # Create a tqdm bar for processing batches within the split
        with tqdm(total=len(dataset_split), desc=f"Processing {split} batches", leave=False,
                  disable=True) as pbar_batches:
            for i in range(0, len(dataset_split), batch_size):
                batch_cache_path = os.path.join(self.cache_dir, f"{split}_batch_{i}.arrow")
                if os.path.exists(batch_cache_path):
                    # Load processed batch from cache
                    logger.debug(f"Loading batch from cache: {batch_cache_path}")
                    cached_batch = load_from_disk(batch_cache_path)
                    yield from cached_batch["text"]
                else:
                    # Process batch and save to cache
                    batch = dataset_split[i: i + batch_size]
                    processed_batch = list(self._preprocess_batch(batch))
                    processed_batch_dataset = Dataset.from_dict({"text": processed_batch})

                    # Disable datasets logging to suppress the progress bar
                    ds_logging.disable_progress_bar()
                    processed_batch_dataset.save_to_disk(batch_cache_path)

                    logger.debug(f"Saved batch to cache: {batch_cache_path}")
                    yield from processed_batch

                pbar_batches.update(batch_size)  # Update the progress bar after processing each batch

    def _preprocess_batch(self, batch: Dict[str, Union[str, List[str]]]) -> Generator[str, None, None]:
        """
        Preprocesses a batch of text examples using vectorized operations and chunking.

        Args:
            batch (Dict[str, Union[str, List[str]]]): A batch of text examples.

        Returns:
            Generator[str, None, None]: A generator yielding preprocessed text data.
        """
        # Flatten text using list comprehension
        texts = [" ".join(text) if isinstance(text, list) else text for text in batch["text"]]

        chunk_size = 100  # Adjust chunk size as needed
        for i in range(0, len(texts), chunk_size):
            text_chunk = texts[i: i + chunk_size]

            # Apply character-level augmentation (vectorized) with reduced probability
            if random.random() < 0.5:  # Apply with 50% probability
                for augmenter in self.char_augmenter:
                    text_chunk = augmenter.augment(text_chunk)

            # Apply word-level augmentation (vectorized) with reduced probability
            if random.random() < 0.3:  # Apply with 30% probability
                for augmenter in self.word_augmenter:
                    text_chunk = augmenter.augment(text_chunk)

            # Apply sentence-level augmentation (vectorized) with reduced probability
            if random.random() < 0.2:  # Apply with 20% probability
                for augmenter in self.sentence_augmenter:
                    text_chunk = augmenter.augment(text_chunk)

            # Remove stopwords and punctuation (vectorized)
            texts_series = pd.Series(text_chunk)
            texts_series = texts_series.str.replace(r"[^\w\s]", "", regex=True)  # Remove punctuation
            texts_series = texts_series.str.split().apply(
                lambda tokens: [token for token in tokens if token.lower() not in stop_words]
            )  # Remove stopwords
            text_chunk = texts_series.str.join(" ").tolist()

            # Apply synonym augmentation (vectorized with probability)
            if random.random() < 0.1:
                text_chunk = self.synonym_augmenter.augment(text_chunk)

            yield from text_chunk  # Yield preprocessed texts from the generator