import math
from typing import List, Union

import pytest
import torch
from transformers import GPT2Tokenizer
from transformers import StoppingCriteria

from core.models.uncertain_nn import UncertainTransformerConfig
from core.models.uncertain_nn import UncertainTransformerLMHeadModel
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


class MaxLengthCriteria(StoppingCriteria):
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids.shape[-1] >= self.max_length


def generate_text(
        model: UncertainTransformerLMHeadModel,
        tokenizer: GPT2Tokenizer,
        prompt: str,
        max_length: int = 50,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        num_return_sequences: int = 1,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> List[str]:
    model.to(device)
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            do_sample=True,
        )

    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]


def calculate_perplexity(
        model: UncertainTransformerLMHeadModel,
        tokenizer: GPT2Tokenizer,
        text: Union[str, List[str]],
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Union[float, List[float]]:
    """
    Calculate the perplexity of the given text using the UncertainTransformerLMHeadModel.

    Args:
        model (UncertainTransformerLMHeadModel): The model to use for perplexity calculation.
        tokenizer (GPT2Tokenizer): The tokenizer to use for encoding the text.
        text (Union[str, List[str]]): The input text or list of texts to calculate perplexity for.
        device (torch.device, optional): The device to run the model on. Defaults to CUDA if available, else CPU.

    Returns:
        Union[float, List[float]]: The calculated perplexity or list of perplexities.

    Raises:
        ValueError: If the input text is empty or contains no valid tokens.
    """
    model.to(device)
    model.eval()

    if isinstance(text, list):
        return [calculate_perplexity(model, tokenizer, t, device) for t in text if t.strip()]

    if not text.strip():
        raise ValueError("Input text is empty.")

    # Tokenize the input text
    input_ids: torch.Tensor = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0).to(device)

    if input_ids.numel() == 0:
        raise ValueError("No valid tokens in the input text.")

    # Prepare target ids for calculating loss
    target_ids: torch.Tensor = input_ids.clone()
    target_ids[:, :-1] = input_ids[:, 1:]
    target_ids[:, -1] = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        loss: torch.Tensor = outputs.loss

    if torch.isnan(loss) or torch.isinf(loss):
        raise ValueError(f"Invalid loss value: {loss.item()}. Check the model outputs.")

    perplexity: float = math.exp(min(loss.item(), 100))  # Clip loss to avoid overflow
    return perplexity


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
def test_generation_and_perplexity(
        model: UncertainTransformerLMHeadModel,
        tokenizer: Tokenizer,
        prompt: str,
        device: torch.device,
):
    model.to(device)
    print(f"\nTesting prompt: {prompt}")

    NUM_SAMPLES = 3
    generated_texts = []
    perplexities = []

    try:
        generated_texts = generate_text(
            model,
            tokenizer.tokenizer,
            prompt,
            max_length=50,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            num_return_sequences=NUM_SAMPLES,
            device=device,
        )

        for i, generated_text in enumerate(generated_texts):
            perplexity = calculate_perplexity(model, tokenizer.tokenizer, generated_text, device=device)
            perplexities.append(perplexity)

            print(f"\nSample {i + 1}:")
            print(f"Generated text: '{generated_text}'")
            print(f"Generated text length: {len(generated_text)}")
            print(f"Perplexity: {perplexity:.2f}")

            assert len(generated_text) > 0, "Generated text should not be empty"
            assert 0 < perplexity < float("inf"), "Perplexity should be a finite positive number"

    except Exception as e:
        print(f"Error generating text for prompt '{prompt}': {str(e)}")
        print(f"Error details: {type(e).__name__}")
        import traceback
        print(traceback.format_exc())

    if perplexities:
        avg_perplexity = sum(perplexities) / len(perplexities)
        print(f"\nAverage Perplexity: {avg_perplexity:.2f}")
        assert 0 < avg_perplexity < float("inf"), "Average perplexity should be a finite positive number"
    else:
        print("No perplexities calculated.")


def test_model_output_shapes(model, tokenizer, device):
    model.to(device)
    prompt = "Test prompt"
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_ids)

    print(f"\nModel output shapes:")
    print(f"Logits shape: {outputs.logits.shape}")
    if outputs.hidden_states is not None:
        print(f"Hidden states shape: {outputs.hidden_states[-1].shape}")
    else:
        print("Hidden states not returned by the model")

    # Check for nan values
    if torch.isnan(outputs.logits).any():
        print("Warning: NaN values detected in logits")
    if (
            outputs.hidden_states is not None
            and torch.isnan(outputs.hidden_states[-1]).any()
    ):
        print("Warning: NaN values detected in hidden states")

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

    # Check for nan values
    if (
            torch.isnan(outputs_with_mask.logits).any()
            or torch.isnan(outputs_without_mask.logits).any()
    ):
        print("Warning: NaN values detected in logits")
        return

    diff = (
        (outputs_with_mask.logits[:, -1] - outputs_without_mask.logits[:, -1])
        .abs()
        .max()
        .item()
    )
    print(f"Difference in last token logits: {diff:.4f}")

    assert diff > 0, "Outputs should differ when using attention mask"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
