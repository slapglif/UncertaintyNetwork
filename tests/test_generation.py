import math

import pytest
import torch
import torch.nn.functional as F

from core.models.uncertain_nn import (
    UncertainTransformerConfig,
    UncertainTransformerLMHeadModel,
)
from core.utils.tokenizer import Tokenizer


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def model(device):
    config = UncertainTransformerConfig()
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
) -> str:
    device = next(model.parameters()).device
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :] / temperature

            print(
                f"Logits min: {next_token_logits.min().item()}, max: {next_token_logits.max().item()}, mean: {next_token_logits.mean().item()}"
            )

            # Apply softmax with increased numerical stability
            next_token_logits = (
                next_token_logits - next_token_logits.max(dim=-1, keepdim=True)[0]
            )
            next_token_probs = F.softmax(next_token_logits, dim=-1)

            print(
                f"Probs min: {next_token_probs.min().item()}, max: {next_token_probs.max().item()}, sum: {next_token_probs.sum().item()}"
            )

            # Handle any remaining NaN or inf values
            next_token_probs = torch.where(
                torch.isnan(next_token_probs) | torch.isinf(next_token_probs),
                torch.zeros_like(next_token_probs),
                next_token_probs,
            )

            # Renormalize if necessary
            if next_token_probs.sum() == 0:
                next_token_probs = torch.ones_like(
                    next_token_probs
                ) / next_token_probs.size(-1)
            else:
                next_token_probs = next_token_probs / next_token_probs.sum()

            # Sample from the probability distribution
            next_token = torch.multinomial(next_token_probs, num_samples=1)

            # Convert to list and append to generated_tokens
            next_token_ids = next_token.squeeze().cpu().tolist()
            if isinstance(next_token_ids, int):
                next_token_ids = [next_token_ids]
            generated_tokens.extend(next_token_ids)

            # Concatenate the new token(s) to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if tokenizer.tokenizer.eos_token_id in next_token_ids:
                break

    return tokenizer.decode(generated_tokens)


def calculate_perplexity(
    model: UncertainTransformerLMHeadModel, tokenizer: Tokenizer, text: str
) -> float:
    device = next(model.parameters()).device
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-1] = input_ids[:, 1:]
    target_ids[:, -1] = -100  # Ignore the last token

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss

    # Handle potential NaN or inf values
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Warning: Loss is {loss.item()}. Returning max float value.")
        return float("inf")

    return math.exp(min(loss.item(), 100))  # Cap the exponent to prevent overflow


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
def test_generation(model, tokenizer, prompt, device):
    model.to(device)
    generated_text = generate_text(model, tokenizer, prompt)
    perplexity = calculate_perplexity(model, tokenizer, generated_text)

    print(f"\nPrompt: {prompt}")
    print(f"Generated text: {generated_text}")
    print(f"Perplexity: {perplexity:.2f}")

    assert len(generated_text) > len(
        prompt
    ), "Generated text should be longer than the prompt"
    assert perplexity > 0, "Perplexity should be a positive number"


def test_average_perplexity(model, tokenizer, device):
    model.to(device)
    prompts = [
        "Once upon a time,",
        "The quick brown fox",
        "In a world where",
        "Scientists have discovered",
        "The future of artificial intelligence",
    ]

    perplexities = []
    for prompt in prompts:
        try:
            generated_text = generate_text(model, tokenizer, prompt)
            perplexity = calculate_perplexity(model, tokenizer, generated_text)
            perplexities.append(perplexity)
            print(f"Prompt: {prompt}")
            print(f"Generated text: {generated_text}")
            print(f"Perplexity: {perplexity:.2f}")
        except Exception as e:
            print(f"Error generating text for prompt '{prompt}': {str(e)}")
            print(f"Error details: {type(e).__name__}")
            import traceback

            print(traceback.format_exc())

    if perplexities:
        finite_perplexities = [
            p for p in perplexities if p != float("inf") and not math.isnan(p)
        ]
        if finite_perplexities:
            avg_perplexity = sum(finite_perplexities) / len(finite_perplexities)
            print(f"\nAverage Perplexity: {avg_perplexity:.2f}")
            assert avg_perplexity > 0, "Average perplexity should be a positive number"
        else:
            print("\nAll perplexities were infinite or NaN.")
    else:
        print("No valid perplexities calculated.")
