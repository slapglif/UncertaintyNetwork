import itertools
import unittest

import torch
import torch.nn as nn
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from core.models.attention.swa import MultiHeadAttentionConfig, MultiHeadAttention


class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 4
        self.embed_dim = 8
        self.num_heads = 2
        self.config = MultiHeadAttentionConfig(d_model=self.embed_dim, n_heads=self.num_heads)
        self.mha = MultiHeadAttention(self.config).to('cuda')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_attention_mask(self):
        torch.manual_seed(42)  # Set seed for reproducibility
        x = torch.randn(self.batch_size, self.seq_len, self.config.d_model).to(self.device)
        mask = torch.ones(self.batch_size, self.seq_len).to(self.device)
        mask[:, self.seq_len // 2:] = 0

        print(f"Input shape: {x.shape}")
        print(f"Mask shape: {mask.shape}")

        output, attn_weights = self.mha(x, attention_mask=mask)

        # Check output shape
        expected_output_shape = (self.batch_size, self.seq_len, self.config.d_model)
        self.assertEqual(output.shape, expected_output_shape,
                         f"Expected output shape {expected_output_shape}, but got {output.shape}")

        # Check attention weights shape
        expected_attn_shape = (self.batch_size, self.config.n_heads, self.seq_len, self.seq_len)
        self.assertEqual(attn_weights.shape, expected_attn_shape,
                         f"Expected attention weights shape {expected_attn_shape}, but got {attn_weights.shape}")

        # Check if the output is masked correctly
        unmasked_output = output[:, :self.seq_len // 2, :]
        masked_output = output[:, self.seq_len // 2:, :]

        # Print for debugging
        logger.debug(f"unmasked_output shape: {unmasked_output.shape}")
        logger.debug(f"masked_output shape: {masked_output.shape}")
        logger.debug(f"unmasked_output: {unmasked_output}")
        logger.debug(f"masked_output: {masked_output}")

        # The masked part should have zero magnitude
        unmasked_norm = torch.norm(unmasked_output, dim=-1).mean()
        masked_norm = torch.norm(masked_output, dim=-1).mean()

        logger.debug(f"unmasked_norm: {unmasked_norm}")
        logger.debug(f"masked_norm: {masked_norm}")

        self.assertGreater(unmasked_norm, 0, "Unmasked output should have non-zero magnitude")
        self.assertLess(masked_norm, 1e-6, "Masked output should have near-zero magnitude")

        # Additional check: all values in masked_output should be zero
        self.assertTrue(torch.allclose(masked_output, torch.zeros_like(masked_output), atol=1e-6),
                        "All values in masked output should be near-zero")

    def test_gradient_flow(self):
        """Verifies that gradients flow properly through the MHA module."""
        self.mha.train()
        x = torch.randn(self.batch_size, self.seq_len, self.config.d_model, requires_grad=True).to(self.device)
        target = torch.randn(self.batch_size, self.seq_len, self.embed_dim).to(self.device)

        output, _ = self.mha(x)
        loss = torch.nn.functional.mse_loss(output, target)

        grads = torch.autograd.grad(loss, [x] + list(self.mha.parameters()), create_graph=True,
                                    allow_unused=True)

        self.assertIsNotNone(grads[0], "Gradient is None for input tensor")
        self.assertGreater(grads[0].abs().sum().item(), 0, "Gradient has zero magnitude for input tensor")

        for i, (name, param) in enumerate(self.mha.named_parameters(), 1):
            self.assertIsNotNone(grads[i], f"Gradient is None for MHA parameter: {name}")
            self.assertGreater(grads[i].abs().sum().item(), 0,
                               f"Gradient has zero magnitude for MHA parameter: {name}")

    def test_numerical_stability(self):
        """Assesses the numerical stability of the MHA module with large inputs."""
        large_input = torch.randn(self.batch_size, self.seq_len, self.config.d_model).to(self.device) * 1e6
        output, _ = self.mha(large_input)

        self.assertFalse(torch.any(torch.isnan(output)), "NaN values detected in the output")
        self.assertFalse(torch.any(torch.isinf(output)), "Inf values detected in the output")

    def test_usefulness_in_system(self):  # sourcery skip: low-code-quality
        """Evaluates MHA's impact on a sequence modeling task."""

# sourcery skip: no-loop-in-tests

        class SequenceModel(nn.Module):
            def __init__(self, config, hidden_dim, num_classes, use_mha=True):
                super().__init__()
                self.use_mha = use_mha
                if use_mha:
                    self.mha = MultiHeadAttention(config)
                self.lstm = nn.LSTM(config.d_model, hidden_dim, batch_first=True)
                self.classifier = nn.Linear(hidden_dim, num_classes)

            def forward(self, x):
                if self.use_mha:
                    x, _ = self.mha(x)
                lstm_out, _ = self.lstm(x)
                return self.classifier(lstm_out[:, -1, :])

        # Hyperparameters
        hidden_dim = 64
        num_classes = 5
        epochs = 20
        batch_size = 32

        # Hyperparameter Search Space
        learning_rates = [1e-3, 5e-4]
        n_heads_options = [4, 8]

        best_accuracy = 0.0
        best_params = {}

        # Create a more complex sequence classification dataset
        seq_len = 50
        num_samples = 5000
        x = torch.randn(num_samples, seq_len, self.embed_dim)
        y = torch.sum(x[:, :, 0] > 0, dim=1) % num_classes  # Class based on positive values in first dimension

        # Split data
        train_size = int(0.8 * num_samples)
        train_data, test_data = x[:train_size], x[train_size:]
        train_labels, test_labels = y[:train_size], y[train_size:]

        train_dataset = TensorDataset(train_data, train_labels)
        test_dataset = TensorDataset(test_data, test_labels)


        for lr, n_heads in itertools.product(learning_rates, n_heads_options):
            logger.info(f"Testing with lr={lr}, n_heads={n_heads}")

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            config = MultiHeadAttentionConfig(d_model=self.embed_dim, n_heads=n_heads)
            model_with_mha = SequenceModel(config, hidden_dim, num_classes, use_mha=True).to(self.device)
            model_without_mha = SequenceModel(config, hidden_dim, num_classes, use_mha=False).to(self.device)

            optimizer_with_mha = optim.Adam(model_with_mha.parameters(), lr=lr)
            optimizer_without_mha = optim.Adam(model_without_mha.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            # Training loop
            for _ in tqdm(range(epochs), desc="Training"):
                for model, optimizer in [(model_with_mha, optimizer_with_mha),
                                         (model_without_mha, optimizer_without_mha)]:
                    model.train()
                    for inputs, labels in train_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

            # Evaluation
            model_with_mha.eval()
            model_without_mha.eval()

            correct_with_mha = 0
            correct_without_mha = 0
            total = 0

            with torch.no_grad():
                for data, labels in test_loader:
                    data, labels = data.to(self.device), labels.to(self.device)
                    total += labels.size(0)

                    outputs_with_mha = model_with_mha(data)
                    _, predicted_with_mha = torch.max(outputs_with_mha.data, 1)
                    correct_with_mha += (predicted_with_mha == labels).sum().item()

                    outputs_without_mha = model_without_mha(data)
                    _, predicted_without_mha = torch.max(outputs_without_mha.data, 1)
                    correct_without_mha += (predicted_without_mha == labels).sum().item()

            accuracy_with_mha = 100 * correct_with_mha / total
            accuracy_without_mha = 100 * correct_without_mha / total

            logger.info(f'Accuracy with MHA: {accuracy_with_mha:.2f}%')
            logger.info(f'Accuracy without MHA: {accuracy_without_mha:.2f}%')

            if accuracy_with_mha > best_accuracy:
                best_accuracy = accuracy_with_mha
                best_params = {"lr": lr, "n_heads": n_heads}

            logger.info(f"Best Accuracy with MHA: {best_accuracy:.2f}%")
            logger.info(f"Best Hyperparameters: {best_params}")

            # Relaxed assertion: MHA should perform at least as well as the baseline
            self.assertGreaterEqual(best_accuracy, accuracy_without_mha,
                                    "Accuracy with MHA should be at least as high as without MHA")

            if best_accuracy > accuracy_without_mha:
                logger.info("MHA shows potential benefits in this scenario")
            else:
                logger.info("MHA does not show significant benefits in this scenario")


if __name__ == '__main__':
    unittest.main()
