from typing import List

import torch
from transformers import GPT2Tokenizer, BatchEncoding, GemmaTokenizer


class Tokenizer:
    """
    A class to handle tokenization using either GPT-2 or Gemma tokenizers.

    This class provides a unified interface for tokenizing text using either the
    pretrained GPT-2 tokenizer or the Gemma tokenizer from KerasNLP. It allows
    for encoding and decoding text, accessing vocabulary, and setting special tokens.

    Attributes:
        tokenizer: The underlying tokenizer object (either GPT2Tokenizer or GemmaTokenizer).
        vocab_size: The vocabulary size of the tokenizer.
        pad_token_id: The ID of the padding token.
        eos_token_id: The ID of the end-of-sequence token.
        bos_token_id: The ID of the beginning-of-sequence token.
    """

    def __init__(
            self, pretrained_tokenizer: str = "gemma-2b", vocab_size: int = 256128
    ):
        """
        Initialize the Tokenizer class.

        Args:
            pretrained_tokenizer (str, optional): The name of the pretrained tokenizer.
                Can be either "gpt2" or a Gemma preset name (e.g., "gemma_2b_en").
                Defaults to "gpt2".
            vocab_size (int, optional): The vocabulary size. Defaults to 50257.
                This is used only for GPT-2; Gemma tokenizers have fixed vocab sizes.
        """
        if pretrained_tokenizer == "gpt2":
            vocab_size = 50257
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                pretrained_tokenizer,  # Initialize with the full GPT-2 vocabulary
                vocab_size=vocab_size,
            )  # Ensure the vocab_size matches the GPT-2 tokenizer
        else:
            # Initialize Gemma tokenizer from the specified preset
            model_name = "google/gemma-7b"
            token = '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'
            self.tokenizer = GemmaTokenizer.from_pretrained(model_name, token=token)

        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
        )  # Set pad token to eos token
        self.vocab_size = vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id

    def encode(
            self, text: str, add_special_tokens: bool = True, **kwargs
    ) -> List[int] | BatchEncoding:
        """
        Encode the given text into a list of token IDs.

        Args:
            text (str): The text to encode.
            add_special_tokens (bool, optional): Whether to add special tokens (BOS, EOS) to the encoded sequence. Defaults to True.
            **kwargs: Additional arguments to pass to the tokenizer's encode_plus method.

        Returns:
            List[int] | BatchEncoding: The encoded token IDs or a BatchEncoding object (depending on the tokenizer).
        """
        return self.tokenizer.encode_plus(
            text, add_special_tokens=add_special_tokens, **kwargs
        )

    def _decode(
            self,
            token_ids: List[int],
            skip_special_tokens: bool = False,
            spaces_between_special_tokens: bool = False,
            **kwargs,
    ) -> str:
        current_sub_text = []
        for id in token_ids:
            if skip_special_tokens and id in self.all_special_ids:
                continue
            if id in self._added_tokens_decoder:
                current_sub_text.append(self.sp_model.decode([id]))
            elif 0 <= id < self.vocab_size:  # Check if token ID is within valid range
                current_sub_text.append(id)
        sub_texts = [self.sp_model.decode(current_sub_text)]
        return "".join(sub_texts)

    def batch_decode(
            self, token_ids_batch: List[torch.Tensor] | torch.Tensor, **kwargs
    ):
        """
        Decode a batch of token IDs back into text.

        Args:
            token_ids_batch (List[torch.Tensor]): A list of token ID tensors to decode.
            **kwargs: Additional arguments to pass to the tokenizer's batch_decode method.

        Returns:
            List[str]: The list of decoded texts.
        """
        if isinstance(token_ids_batch, torch.Tensor):
            token_ids_batch = [token_ids_batch]

        decoded_texts = []
        for ids_list in token_ids_batch:
            if isinstance(ids_list, torch.Tensor):
                ids_list = ids_list.tolist()
            flattened_ids = [id for sublist in ids_list for id in sublist]
            decoded_texts.append(self.tokenizer.decode(flattened_ids, **kwargs))
        return decoded_texts

    @property
    def pad_token(self):
        """
        Get the padding token.
        """
        return self.tokenizer.pad_token

    @pad_token.setter
    def pad_token(self, value):
        """
        Set the padding token.
        """
        self.tokenizer.pad_token = value
        self.pad_token_id = self.tokenizer.pad_token_id

    @property
    def eos_token(self):
        """
        Get the end-of-sequence token.
        """
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
