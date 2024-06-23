# core/utils/tokenizer.py

from transformers import GPT2Tokenizer

class Tokenizer:
    def __init__(self, pretrained_tokenizer: str = "gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, token_ids):
        return self.tokenizer.convert_ids_to_tokens(token_ids)

    def get_special_tokens_mask(self, token_ids, already_has_special_tokens=False):
        return self.tokenizer.get_special_tokens_mask(token_ids, already_has_special_tokens)

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    @staticmethod
    def from_pretrained(pretrained_tokenizer="gpt2"):
        return Tokenizer(pretrained_tokenizer)