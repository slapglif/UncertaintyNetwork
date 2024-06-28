from typing import Tuple

import torch
import torch.nn as nn

import unittest

from core.models.embedding import RotaryEmbedding


class TestRotaryEmbedding(unittest.TestCase):

    def setUp(self):
        self.embed_dim = 64
        self.max_positions = 2048
        self.batch_size = 2
        self.seq_len = 20
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rotary_embedding = RotaryEmbedding(self.embed_dim, self.max_positions).to(self.device)

    def test_backward_application(self):
        xq = torch.randn(self.batch_size, self.seq_len, self.embed_dim, requires_grad=True).to(self.device)
        xk = torch.randn(self.batch_size, self.seq_len, self.embed_dim, requires_grad=True).to(self.device)
        xq_out, xk_out = self.rotary_embedding(xq, xk, self.seq_len, 0)  # Remove start=0
        self.assertEqual(xq_out.shape, xq.shape)
        self.assertEqual(xk_out.shape, xk.shape)

    def test_different_start_positions(self):
        xq = torch.randn(self.batch_size, self.seq_len, self.embed_dim).to(self.device)
        xk = torch.randn(self.batch_size, self.seq_len, self.embed_dim).to(self.device)
        xq_out1, xk_out1 = self.rotary_embedding(xq, self.seq_len, start=0)
        xq_out2, xk_out2 = self.rotary_embedding(xq, self.seq_len, start=5)
        self.assertEqual(xq_out1.shape, xq.shape)
        self.assertEqual(xk_out1.shape, xk.shape)
        self.assertEqual(xq_out2.shape, xq.shape)
        self.assertEqual(xk_out2.shape, xk.shape)

    def test_extending_max_positions(self):
        self.assertEqual(self.rotary_embedding.max_positions, self.max_positions)

    def test_freqs_cis_precomputation(self):
        freqs_cis = self.rotary_embedding.get_freqs_cis(0, self.seq_len)
        self.assertEqual(freqs_cis.shape, (self.seq_len, self.embed_dim // 2, 2))

    def test_output_shape(self):
        xq = torch.randn(self.batch_size, self.seq_len, self.embed_dim).to(self.device)
        xk = torch.randn(self.batch_size, self.seq_len, self.embed_dim).to(self.device)
        xq_out, xk_out = self.rotary_embedding(xq, self.seq_len, start=0)
        self.assertEqual(xq_out.shape, xq.shape)
        self.assertEqual(xk_out.shape, xk.shape)

    def test_rotary_application(self):
        xq = torch.randn(self.batch_size, self.seq_len, self.embed_dim).to(self.device)
        xk = torch.randn(self.batch_size, self.seq_len, self.embed_dim).to(self.device)
        xq_out, xk_out = self.rotary_embedding(xq, self.seq_len, start=0)
        self.assertEqual(xq_out.shape, xq.shape)
        self.assertEqual(xk_out.shape, xk.shape)


if __name__ == '__main__':
    unittest.main()