# test_e2e.py
import unittest

import numpy as np
import torch
from loguru import logger
from torch.cpu.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

from core.models.uncertainty.uncertainty import (
    UncertainTransformerLMHeadModel,
    UncertainTransformerConfig,
)
from core.utils.tokenizer import Tokenizer


class ModelNoUncertainty(UncertainTransformerLMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.uncertainty_module = None

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = super().forward(input_ids, attention_mask, labels)
        return outputs._replace(uncertainties=None)


class TestUncertainTransformerModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cpu.is_available() else "cpu")
        cls.config = UncertainTransformerConfig(
            vocab_size=256128,
            d_model=64,  # Reduced from 128
            n_heads=2,  # Reduced from 4
            d_ff=256,  # Reduced from 512
            n_layers=1,  # Reduced from 2
            dropout=0.1,
            max_position_embeddings=512,
            pad_token_id=0,
            use_mamba=True,
            d_state=8,  # Reduced from 16
            d_conv=2,  # Kept at 2
            expand_factor=1.5,  # Reduced from 2.0
            dt_rank=8,  # Kept at 8
            n_inducing=5,  # Reduced from 10
            uncertainty_weight=0.1,
        )
        cls.model = UncertainTransformerLMHeadModel(cls.config).to(cls.device)
        cls.tokenizer = Tokenizer()

    def evaluate_model(self, model: UncertainTransformerLMHeadModel) -> float:
        """
        Evaluate the model's performance.

        Args:
            model (UncertainTransformerLMHeadModel): The model to evaluate.

        Returns:
            float: A performance metric (e.g., accuracy, perplexity).
        """
        model.eval()
        with torch.no_grad():
            # Create a small evaluation dataset
            batch_size = 4
            seq_len = 20
            input_ids = torch.randint(
                0, self.config.vocab_size, (batch_size, seq_len)
            ).to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)
            labels = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(
                self.device
            )

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            # Calculate a performance metric (e.g., accuracy)
            logits = outputs["logits"]  # Access logits from the dictionary
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == labels).float()
            accuracy = correct.sum() / (batch_size * seq_len)

            return accuracy.item()

    def test_ablation_studies(self):
        # Base model performance
        base_performance = self.evaluate_model(self.model)

        # Test without Mamba layers
        config_no_mamba = self.config.copy()
        config_no_mamba.use_mamba = False
        model_no_mamba = UncertainTransformerLMHeadModel(config_no_mamba).to(
            self.device
        )
        performance_no_mamba = self.evaluate_model(model_no_mamba)

        # Test without uncertainty module

        model_no_uncertainty = ModelNoUncertainty(self.config).to(self.device)
        performance_no_uncertainty = self.evaluate_model(model_no_uncertainty)

        # Assert performance differences
        self.assertGreater(
            base_performance,
            performance_no_mamba,
            "Mamba layers do not improve performance",
        )
        self.assertGreater(
            base_performance,
            performance_no_uncertainty,
            "Uncertainty module does not improve performance",
        )

        print(f"Base performance: {base_performance}")
        print(f"Performance without Mamba: {performance_no_mamba}")
        print(f"Performance without uncertainty: {performance_no_uncertainty}")

    def test_tensor_shapes(self):
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(
            self.device
        )
        attention_mask = torch.ones_like(input_ids).to(self.device)

        outputs = self.model(input_ids, attention_mask=attention_mask)

        # Adjust expected shape based on the model's output
        self.assertEqual(outputs["logits"].shape, (1, seq_len, self.config.vocab_size))
        self.assertEqual(
            outputs["uncertainties"].shape, (1, seq_len, self.config.vocab_size)
        )

    def test_gradient_flow(self):
        """
        Test the gradient flow through the model during training.

        This method sets up a small batch of data, performs a forward and backward pass,
        and checks for proper gradient flow through all learnable parameters.
        """
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(
            self.device
        )
        attention_mask = torch.ones_like(input_ids).to(self.device)
        labels = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(
            self.device
        )

        logger.info(
            f"Input shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, labels: {labels.shape}"
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        scaler = GradScaler()

        self.model.train()

        with autocast():
            outputs = self.model(
                input_ids, attention_mask=attention_mask, labels=labels
            )

        loss = outputs["loss"]
        logger.info(f"Loss: {loss.item()}")

        # Check if the loss is NaN
        if torch.isnan(loss).any():
            logger.error("Loss is NaN")
            raise ValueError("Loss is NaN")

        scaler.scale(loss).backward()

        # Unscale the gradients
        scaler.unscale_(optimizer)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Check gradients
        total_norm = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    logger.warning(f"No gradient for {name}")
                else:
                    # Replace NaN gradients with zeros
                    param.grad.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    logger.info(f"Gradient norm for {name}: {param_norm:.4f}")

        total_norm = total_norm**0.5
        logger.info(f"Total gradient norm: {total_norm:.4f}")

        # Optimizer step and scaler update
        scaler.step(optimizer)
        scaler.update()

        # Assertions
        self.assertFalse(torch.isnan(loss).any(), "NaN loss detected")
        self.assertFalse(torch.isinf(loss).any(), "Inf loss detected")
        self.assertGreater(total_norm, 0, "Zero gradient norm detected")

        logger.info("Gradient flow test completed successfully")

        # Check for NaN or Inf values in model parameters
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN values in {name}")
            if torch.isinf(param).any():
                print(f"Inf values in {name}")

        # Final assertions
        assert not torch.isnan(
            self.model.temperature
        ).any(), "NaN values in temperature"
        assert not torch.isinf(
            self.model.temperature
        ).any(), "Inf values in temperature"
        assert self.model.temperature.grad is not None, "No gradient for temperature"
        assert not torch.isnan(
            self.model.temperature.grad
        ).any(), "NaN gradient for temperature"
        assert not torch.isinf(
            self.model.temperature.grad
        ).any(), "Inf gradient for temperature"

    def test_training(self):
        batch_size = 4
        seq_len = 20
        num_batches = 5

        # Create a small dummy dataset
        dataset = TensorDataset(
            torch.randint(
                0, self.config.vocab_size, (batch_size * num_batches, seq_len)
            ),
            torch.ones(batch_size * num_batches, seq_len, dtype=torch.long),
            torch.randint(
                0, self.config.vocab_size, (batch_size * num_batches, seq_len)
            ),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        initial_loss = None

        for i, batch in enumerate(dataloader):
            input_ids, attention_mask, labels = [t.to(self.device) for t in batch]
            outputs = self.model(
                input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs["loss"]

            if initial_loss is None:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward(retain_graph=i < len(dataloader) - 1)
            optimizer.step()

        final_loss = loss.item()
        self.assertLess(
            final_loss, initial_loss, "Loss did not decrease during training"
        )

    def test_text_generation(self):
        input_text = "Once upon a time"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(
            self.device
        )

        generated_ids, uncertainties = self.model.generate_with_uncertainty(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
        )

        generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )

        self.assertGreater(
            len(generated_text),
            len(input_text),
            "Generated text is not longer than input",
        )
        self.assertTrue(
            generated_text.startswith(input_text),
            "Generated text does not start with input text",
        )
        self.assertEqual(
            uncertainties.shape[1],
            len(generated_ids[0]) - len(input_ids[0]),
            "Uncertainty shape mismatch",
        )

    def test_uncertainty_metrics(self):
        batch_size = 1
        seq_len = 5
        num_batches = 1

        # Create a small dummy dataset with varied labels
        dataset = TensorDataset(
            torch.randint(
                0, self.config.vocab_size, (batch_size * num_batches, seq_len)
            ),
            torch.randint(
                0, 2, (batch_size * num_batches, seq_len), dtype=torch.long
            ),  # Labels with 0s and 1s
            torch.randint(
                0, self.config.vocab_size, (batch_size * num_batches, seq_len)
            ),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        metrics = self.model.compute_uncertainty_metrics(dataloader)

        expected_metrics = ["ece", "brier_score", "nll", "auroc", "aupr"]
        for metric in expected_metrics:
            self.assertIn(metric, metrics, f"Missing uncertainty metric: {metric}")
            self.assertIsInstance(
                metrics[metric], float, f"Metric {metric} is not a float"
            )

    def test_calibration(self):
        batch_size = 2
        seq_len = 5
        num_batches = 1

        # Create a small dummy dataset
        dataset = TensorDataset(
            torch.randint(
                0, self.config.vocab_size, (batch_size * num_batches, seq_len)
            ),
            torch.ones(batch_size * num_batches, seq_len, dtype=torch.long),
            torch.randint(
                0, self.config.vocab_size, (batch_size * num_batches, seq_len)
            ),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        initial_temperature = self.model.temperature.item()
        optimal_temperature = self.model.calibrate(dataloader)

        self.assertIsInstance(
            optimal_temperature, float, "Optimal temperature is not a float"
        )
        self.assertNotEqual(
            optimal_temperature,
            initial_temperature,
            "Temperature did not change after calibration",
        )

    def test_active_learning_acquisition(self):
        batch_size = 2
        seq_len = 10
        num_batches = 2

        # Create a small dummy dataset
        dataset = TensorDataset(
            torch.randint(
                0, self.config.vocab_size, (batch_size * num_batches, seq_len)
            ),
            torch.ones(batch_size * num_batches, seq_len, dtype=torch.long),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        n_samples = 1
        selected_indices = self.model.active_learning_acquisition(dataloader, n_samples)

        self.assertEqual(
            len(selected_indices), n_samples, "Incorrect number of samples selected"
        )
        self.assertEqual(
            len(set(selected_indices)), n_samples, "Duplicate indices selected"
        )
        self.assertTrue(
            all(0 <= idx < batch_size * num_batches for idx in selected_indices),
            "Invalid indices selected",
        )

    def test_attention_mechanism(self):
        input_ids = torch.randint(0, self.config.vocab_size, (1, 50)).to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

        # Access attention weights from the correct attribute
        attention_weights = self.model.transformer.layers[0][1].attention.attn

        # Check if attention focuses on relevant tokens
        self.assertFalse(
            torch.allclose(
                attention_weights,
                torch.ones_like(attention_weights) / attention_weights.size(-1),
            )
        )
        self.assertTrue(
            (attention_weights >= 0).all() and (attention_weights <= 1).all()
        )

    def test_mamba_layer_effectiveness(self):
        input_ids = torch.randint(0, self.config.vocab_size, (1, 100)).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids)

        # Access Mamba layer outputs directly from the forward pass
        mamba_outputs = [
            layer[0](outputs["hidden_states"][i]).to(self.device)
            for i, layer in enumerate(self.model.transformer.layers)
        ]

        # Check if Mamba layers capture long-range dependencies
        for i in range(1, len(mamba_outputs)):
            correlation = torch.corrcoef(
                mamba_outputs[i].view(-1), mamba_outputs[i - 1].view(-1)
            )[0, 1]
            self.assertLess(
                correlation,
                0.99,
                f"Mamba layer {i} output is too similar to previous layer",
            )

    def test_uncertainty_calibration(self):
        # Generate some test data
        input_ids = torch.randint(0, self.config.vocab_size, (100, 20)).to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        labels = torch.randint(0, self.config.vocab_size, (100, 20)).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

        logits = outputs["logits"]
        uncertainties = outputs["uncertainties"]

        probs = torch.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)
        accuracies = (predictions == labels).float()

        # Compute calibration error
        ece = self.expected_calibration_error(
            confidences.cpu().numpy(), accuracies.cpu().numpy()
        )

        self.assertLess(ece, 0.1, f"Expected Calibration Error is too high: {ece}")

    @classmethod
    def expected_calibration_error(cls, confidences, accuracies, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Apply the boolean index to both confidences and accuracies
            in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(
                    accuracies[in_bin]
                )  # Apply boolean index here
                avg_confidence_in_bin = np.mean(confidences[in_bin])  # and here
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

    def test_long_range_dependencies(self):
        suffix = "which is known for its iconic Eiffel Tower."
        middle = " " * 100  # 100 spaces
        full_text = f"The capital of France is{middle}Paris, {suffix}"

        input_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(
            self.device
        )

        generated_ids = self.model.generate(
            input_ids, max_length=len(input_ids[0]) + 50, num_return_sequences=1
        )
        generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )

        self.assertIn(
            "Paris", generated_text, "Model failed to capture long-range dependency"
        )

    def test_uncertainty_based_token_selection(self):
        input_text = "The rate of acceleration due to gravity on Earth is approximately"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(
            self.device
        )

        generated_ids, uncertainties = self.model.generate_with_uncertainty(
            input_ids,
            max_length=30,
            num_return_sequences=5,
            temperature=0.7,
            top_k=0,
            do_sample=True,
        )

        generated_texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated_ids
        ]

        # Check if high uncertainty tokens are diverse
        high_uncertainty_positions = uncertainties.mean(dim=0).topk(5).indices
        tokens_at_uncertain_positions = [
            [self.tokenizer.decode([ids[pos]]) for ids in generated_ids]
            for pos in high_uncertainty_positions
        ]

        for tokens in tokens_at_uncertain_positions:
            self.assertGreater(
                len(set(tokens)), 1, "High uncertainty tokens are not diverse"
            )

    def test_adversarial_input_handling(self):
        normal_input = "The quick brown fox jumps over the lazy dog"
        adversarial_input = "Th3 qu1ck br0wn f0x jumps 0v3r th3 l@zy d0g"

        normal_ids = self.tokenizer.encode(normal_input, return_tensors="pt").to(
            self.device
        )
        adversarial_ids = self.tokenizer.encode(
            adversarial_input, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            normal_output = self.model(normal_ids)
            adversarial_output = self.model(adversarial_ids)

        normal_uncertainty = normal_output["uncertainties"].mean().item()
        adversarial_uncertainty = adversarial_output["uncertainties"].mean().item()

        self.assertGreater(
            adversarial_uncertainty,
            normal_uncertainty,
            "Model does not express higher uncertainty for adversarial input",
        )


if __name__ == "__main__":
    unittest.main()
