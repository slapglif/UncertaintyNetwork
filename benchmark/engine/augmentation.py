# augmentation.py

import random
import nltk
from nltk.corpus import wordnet
from typing import List, Callable
import torch
from transformers import PreTrainedTokenizer

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def synonym_replacement(text: str, n: int = 1) -> str:
    words = nltk.word_tokenize(text)
    new_words = words.copy()
    random_word_list = list({word for word in words if word.isalnum()})
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_words)


def get_synonyms(word: str) -> set:
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return synonyms


def random_insertion(text: str, n: int = 1) -> str:
    words = nltk.word_tokenize(text)
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return ' '.join(new_words)


def add_word(words: List[str]):
    synonyms = []
    counter = 0
    while not synonyms:
        random_word = random.choice(words)
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = random.choice(list(synonyms))
    random_idx = random.randint(0, len(words) - 1)
    words.insert(random_idx, random_synonym)


def random_swap(text: str, n: int = 1) -> str:
    words = nltk.word_tokenize(text)
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return ' '.join(new_words)


def swap_word(words: List[str]) -> List[str]:
    random_idx_1 = random.randint(0, len(words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(words) - 1)
        counter += 1
        if counter > 3:
            return words
    words[random_idx_1], words[random_idx_2] = words[random_idx_2], words[random_idx_1]
    return words


def random_deletion(text: str, p: float = 0.1) -> str:
    words = nltk.word_tokenize(text)
    if len(words) == 1:
        return words[0]
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)
    if not new_words:
        rand_int = random.randint(0, len(words) - 1)
        return words[rand_int]
    return ' '.join(new_words)


def apply_augmentations(text: str, augmentation_funcs: List[Callable], num_augmentations: int = 1) -> List[str]:
    augmented_texts = []
    for _ in range(num_augmentations):
        augmented_text = text
        for func in augmentation_funcs:
            augmented_text = func(augmented_text)
        augmented_texts.append(augmented_text)
    return augmented_texts


def token_augmentation(input_ids: torch.Tensor, tokenizer: PreTrainedTokenizer,
                       mask_prob: float = 0.15) -> torch.Tensor:
    # Randomly mask tokens
    mask = torch.rand(input_ids.shape) < mask_prob
    masked_input_ids = input_ids.clone()
    masked_input_ids[mask] = tokenizer.mask_token_id
    return masked_input_ids


# Example usage
augmentation_pipeline = [
    lambda x: synonym_replacement(x, n=2),
    lambda x: random_insertion(x, n=1),
    lambda x: random_swap(x, n=1),
    lambda x: random_deletion(x, p=0.1)
]
