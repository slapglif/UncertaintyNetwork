import unittest

import torch
from torch import GradScaler
from torch.cuda.amp import autocast
from torch.utils.data import TensorDataset, DataLoader

from core.models.uncertainty.uncertainty import UncertainTransformerConfig, UncertainTransformerLMHeadModel
from core.utils.tokenizer import Tokenizer


class TestUncertainTransformerLMHeadModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda"
        cls.config = UncertainTransformerConfig(
            vocab_size=256128,
            d_model=64,
            n_heads=2,
            d_ff=128,
            n_layers=1,
            dropout=0.1,
            max_position_embeddings=512,
            pad_token_id=0,
            use_mamba=True,
            d_state=8,
            d_conv=2,
            expand_factor=1.5,
            dt_rank=8,
            n_inducing=5,
            uncertainty_weight=0.1,
        )
        cls.model = UncertainTransformerLMHeadModel(cls.config).to(cls.device)
        cls.tokenizer = Tokenizer()
        cls.batch_size = 4
        cls.seq_len = 20
        cls.test_data = TensorDataset(
            torch.randint(0, cls.config.vocab_size, (cls.batch_size, cls.seq_len)),
            torch.ones(cls.batch_size, cls.seq_len, dtype=torch.long),
            torch.randint(0, cls.config.vocab_size, (cls.batch_size, cls.seq_len)),
        )
        cls.dataloader = DataLoader(cls.test_data, batch_size=cls.batch_size)

    def test_output_shape(self):
        """Checks the output shape of the model."""
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len)).to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        self.assertEqual(outputs["logits"].shape, (self.batch_size, self.seq_len, self.config.vocab_size))
        self.assertEqual(outputs["uncertainties"].shape, (self.batch_size, self.seq_len, self.config.vocab_size))

    def test_loss_calculation(self):
        """Verifies that the loss is calculated correctly."""
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len)).to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        labels = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len)).to(self.device)
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        self.assertTrue(outputs["loss"].item() > 0)

    def test_gradient_flow(self):
        """Ensures that gradients flow through all parameters."""
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len)).to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        labels = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len)).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        scaler = GradScaler()

        self.model.train()

        with autocast():
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # sourcery skip: no-loop-in-tests
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient is None for parameter {name}")

        scaler.step(optimizer)
        scaler.update()

    def test_generation_with_uncertainty(self):
        """Tests the generate_with_uncertainty method."""
        input_text = "Once upon a time"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        generated_ids, uncertainties = self.model.generate_with_uncertainty(
            input_ids, max_length=50, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.95, do_sample=True
        )
        self.assertEqual(generated_ids.shape, (1, 50))
        self.assertEqual(uncertainties.shape, (1, 47, 50257))

    def test_uncertainty_metrics(self):
        """Checks the computation of uncertainty metrics."""
        metrics = self.model.compute_uncertainty_metrics(self.dataloader)
        expected_metrics = ["ece", "brier_score", "nll", "auroc", "aupr"]
        # sourcery skip: no-loop-in-tests
        for metric in expected_metrics:
            self.assertIn(metric, metrics, f"Missing uncertainty metric: {metric}")
            self.assertIsInstance(metrics[metric], float, f"Metric {metric} is not a float")

    def test_calibration(self):
        """Tests the model's calibration functionality."""
        initial_temperature = self.model.temperature.item()
        optimal_temperature = self.model.calibrate(self.dataloader)
        self.assertIsInstance(optimal_temperature, float, "Optimal temperature is not a float")
        self.assertNotEqual(optimal_temperature, initial_temperature, "Temperature did not change after calibration")

    def test_active_learning_acquisition(self):
        """Evaluates the active learning acquisition function."""
        n_samples = 2
        selected_indices = self.model.active_learning_acquisition(self.dataloader, n_samples)
        self.assertEqual(len(selected_indices), n_samples, "Incorrect number of samples selected")
        self.assertTrue(all(0 <= idx < self.batch_size for idx in selected_indices), "Invalid indices selected")
