from typing import List

import torch
from transformers import GPT2Tokenizer


class Tokenizer:
    def __init__(self, pretrained_tokenizer: str = "gpt2", vocab_size: int = 1000):
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = vocab_size - 2
        self.bos_token_id = vocab_size - 1

    def encode(self, text: str, **kwargs):
        """
        Encode the given text into token IDs.

        Args:
            text (str): The text to encode.
            **kwargs: Additional arguments to pass to the tokenizer's encode method.

        Returns:
            torch.Tensor: The encoded token IDs as a tensor.
        """
        encoded = self.tokenizer.encode(text, add_special_tokens=True, **kwargs)
        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor, **kwargs):
        """
        Decode the given token IDs back into text.

        Args:
            token_ids (torch.Tensor): The token IDs to decode.
            **kwargs: Additional arguments to pass to the tokenizer's decode method.

        Returns:
            str: The decoded text.
        """
        return self.tokenizer.decode(token_ids.tolist(), **kwargs)

    def batch_decode(self, token_ids_batch: List[torch.Tensor], **kwargs):
        """
        Decode a batch of token IDs back into text.

        Args:
            token_ids_batch (List[torch.Tensor]): A list of token ID tensors to decode.
            **kwargs: Additional arguments to pass to the tokenizer's batch_decode method.

        Returns:
            List[str]: The list of decoded texts.
        """
        return self.tokenizer.batch_decode([ids.tolist() for ids in token_ids_batch], **kwargs)


    @property
    def pad_token(self):
        return self.tokenizer.pad_token

    @pad_token.setter
    def pad_token(self, value):
        self.tokenizer.pad_token = value
        self.pad_token_id = self.tokenizer.pad_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    def tokenize(self, text: str, **kwargs):
        """
        Tokenize the given text.

        Args:
            text (str): The text to tokenize.
            **kwargs: Additional arguments to pass to the tokenizer's tokenize method.

        Returns:
            List[str]: The list of tokens.
        """
        return self.tokenizer.tokenize(text, **kwargs)

    def convert_tokens_to_ids(self, tokens: list):
        """
        Convert tokens to their corresponding IDs.

        Args:
            tokens (list): The list of tokens to convert.

        Returns:
            List[int]: The list of token IDs.
        """
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, token_ids: list):
        """
        Convert token IDs to their corresponding tokens.

        Args:
            token_ids (list): The list of token IDs to convert.

        Returns:
            List[str]: The list of tokens.
        """
        return self.tokenizer.convert_ids_to_tokens(token_ids)

    def get_vocab(self):
        """
        Get the vocabulary of the tokenizer.

        Returns:
            dict: The vocabulary dictionary.
        """
        return self.tokenizer.get_vocab()

    @staticmethod
    def from_pretrained(pretrained_tokenizer="gpt2"):
        """
        Create a Tokenizer instance from a pretrained tokenizer.

        Args:
            pretrained_tokenizer (str): The name of the pretrained tokenizer to use.

        Returns:
            Tokenizer: An instance of the Tokenizer class.
        """
        return Tokenizer(pretrained_tokenizer)

    def __call__(self, text: str, **kwargs):
        """
        Tokenize the input text and return the encoded input.

        Args:
            text (str): The input text to tokenize.
            **kwargs: Additional arguments to pass to the tokenizer's __call__ method.

        Returns:
            dict: A dictionary containing the encoded input.
        """
        return self.tokenizer(text, **kwargs)
