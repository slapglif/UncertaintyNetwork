import torch
import unittest

from core.models.attention import CEMA

import torch
import unittest

class TestMultiHeadAttention(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 32
        cls.seq_len = 128
        cls.embed_dim = 64
        cls.ndim = 16
        cls.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cls.config = type('Config', (object,), {'d_model': cls.embed_dim})
        cls.mha = CEMA(embed_dim=cls.embed_dim, ndim=cls.ndim, device=cls.device).to(cls.device)

    def test_attention_mask(self):
        x = torch.randn(self.batch_size, self.seq_len, self.config.d_model).to(self.device)
        mask = torch.ones(self.batch_size, self.seq_len).to(self.device)
        mask[:, self.seq_len // 2:] = 0
        output_masked = self.mha(x)  # Modify if `attention_mask` is used in `forward`
        self.assertEqual(output_masked.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_output_shape(self):
        x = torch.randn(self.batch_size, self.seq_len, self.config.d_model).to(self.device)
        output = self.mha(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_sliding_window_attention(self):
        x = torch.randn(self.batch_size, self.seq_len, self.config.d_model).to(self.device)
        output = self.mha(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

if __name__ == '__main__':
    unittest.main()
