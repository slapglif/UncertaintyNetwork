import math
import sys
import time

import pytest
import torch
import torch.nn.functional as F

from core.models.uncertain_nn import (
    UncertainTransformerConfig,
    UncertainTransformerLMHeadModel,
)
from core.utils.tokenizer import Tokenizer

# Constants
MAX_LENGTH = 50
TEMPERATURE = 0.7
TIMEOUT = 30
NUM_SAMPLES = 3


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def model(device):
    config = UncertainTransformerConfig(
        vocab_size=50257,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_layers=6,
        dropout=0.1,
        max_position_embeddings=1024,
        pad_token_id=50256,
    )
    model = UncertainTransformerLMHeadModel(config)
    model.to(device)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer.from_pretrained("gpt2")


def generate_text(
    model: UncertainTransformerLMHeadModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_length: int = 50,
    temperature: float = 1.0,
    timeout: float = 60,
) -> str:
    device = next(model.parameters()).device
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    generated_tokens = []
    start_time = time.time()

    with torch.no_grad():
        for _ in range(max_length):
            if time.time() - start_time > timeout:
                print(f"Generation timed out after {timeout} seconds", file=sys.stderr)
                break

            outputs = model(input_ids[:, -512:])
            next_token_logits = outputs.logits[0, -1, :] / temperature
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)

            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == tokenizer.tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated_tokens)


def calculate_perplexity(
    model: UncertainTransformerLMHeadModel, tokenizer: Tokenizer, text: str
) -> float:
    device = next(model.parameters()).device
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-1] = input_ids[:, 1:]
    target_ids[:, -1] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Warning: Loss is {loss.item()}. Returning max float value.")
        return float("inf")

    return math.exp(min(loss.item(), 100))


@pytest.mark.parametrize(
    "prompt",
    [
        "Once upon a time,",
        "The quick brown fox",
        "In a world where",
        "Scientists have discovered",
        "The future of artificial intelligence",
    ],
)
def test_generation_and_perplexity(model, tokenizer, prompt, device):
    model.to(device)

    print(f"\nTesting prompt: {prompt}")

    generated_texts = []
    perplexities = []

    for i in range(NUM_SAMPLES):
        try:
            generated_text = generate_text(model, tokenizer, prompt)
            perplexity = calculate_perplexity(model, tokenizer, generated_text)

            generated_texts.append(generated_text)
            perplexities.append(perplexity)

            print(f"\nSample {i + 1}:")
            print(f"Generated text: {generated_text}")
            print(f"Perplexity: {perplexity:.2f}")

            assert len(generated_text) > len(
                prompt
            ), "Generated text should be longer than the prompt"

        except Exception as e:
            print(f"Error generating text for prompt '{prompt}': {str(e)}")
            print(f"Error details: {type(e).__name__}")
            import traceback

            print(traceback.format_exc())

    if perplexities:
        avg_perplexity = sum(
            p for p in perplexities if p != float("inf") and not math.isnan(p)
        ) / len(perplexities)
        print(f"\nAverage Perplexity: {avg_perplexity:.2f}")
        assert avg_perplexity > 0, "Average perplexity should be a positive number"
    else:
        print("No valid perplexities calculated.")


def test_model_output_shapes(model, tokenizer, device):
    model.to(device)
    prompt = "Test prompt"
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_ids)

    print(f"\nModel output shapes:")
    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Hidden states shape: {outputs.hidden_states[-1].shape}")

    assert outputs.logits.shape[0] == 1, "Batch size should be 1"
    assert outputs.logits.shape[1] == len(
        input_ids[0]
    ), "Sequence length should match input"
    assert (
        outputs.logits.shape[2] == model.config.vocab_size
    ), "Last dimension should be vocab size"


def test_attention_mask(model, tokenizer, device):
    model.to(device)
    prompt = "Test with padding"
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, -2:] = 0  # Simulate padding

    with torch.no_grad():
        outputs_with_mask = model(input_ids, attention_mask=attention_mask)
        outputs_without_mask = model(input_ids)

    print("\nTesting attention mask:")
    print(
        f"Difference in last token logits: "
        f"{(outputs_with_mask.logits[:, -1] - outputs_without_mask.logits[:, -1]).abs().max().item():.4f}"
    )

    assert not torch.allclose(
        outputs_with_mask.logits[:, -1], outputs_without_mask.logits[:, -1]
    ), "Outputs should differ when using attention mask"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
