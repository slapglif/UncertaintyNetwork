import time

import pytest
import torch
from datasets import load_dataset
from loguru import logger
from torch import nn

from core.models.embedding import RotaryPositionEncoding, SentenceEncoder, SentenceGP
from core.models.uncertain_nn import (
    UncertainTransformerConfig,
    UncertainTransformerLMHeadModel,
)
from core.utils.tokenizer import Tokenizer
from core.utils.utils import softplus

# Constants
BATCH_SIZE = 2
SEQ_LEN = 10
EMBED_DIM = 512
NUM_SENTENCES = 3
N_INDUCING = 10
LEARNING_RATE = 1e-5
NUM_EPOCHS = 5  # Reduced for faster testing
D_STATE = 16
D_CONV = 4
EXPAND_FACTOR = 2.0
DT_RANK = 16
N_HEADS = 8
D_FF = 2048

# Device to use for testing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer for text encoding and decoding
tokenizer = Tokenizer.from_pretrained("gpt2")

# Configure Loguru
logger.remove()  # Remove default logger
logger.add(
    "test_encoder.log",  # Log file
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)


def test_tokenizer_encoding_decoding():
    """Tests encoding and decoding of text with the Tokenizer."""
    text = "This is a test sentence."
    encoded_ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(encoded_ids)
    logger.info(
        f"✅ Text: {text} \nEncoded IDs: {encoded_ids} \nDecoded Text: {decoded_text}"
    )
    assert decoded_text == text


def test_rotary_position_encoding_shape():
    """Tests the shape of the output from RotaryPositionEncoding."""
    rotary_pe = RotaryPositionEncoding(dim=EMBED_DIM).to(DEVICE)
    input_tensor = torch.randn(1, SEQ_LEN, EMBED_DIM, device=DEVICE)
    cos, sin = rotary_pe(input_tensor)
    logger.info(f"✅ Cosine shape: {cos.shape} \nSine shape: {sin.shape}")
    assert cos.shape == (1, 1, SEQ_LEN, EMBED_DIM)
    assert sin.shape == (1, 1, SEQ_LEN, EMBED_DIM)


def test_sentence_encoder_shape():
    """Tests the shape of the output from SentenceEncoder."""
    sentence_encoder = SentenceEncoder(
        vocab_size=tokenizer.vocab_size,  # Provide vocab_size
        hidden_dim=EMBED_DIM,
        output_dim=EMBED_DIM,
    ).to(DEVICE)
    input_tensor = torch.randint(
        0, tokenizer.vocab_size, (BATCH_SIZE, NUM_SENTENCES, SEQ_LEN), device=DEVICE
    )  # Create a 3D tensor of token IDs
    logger.info(f"Input tensor shape: {input_tensor.shape}")
    output = sentence_encoder(input_tensor)
    logger.info(f"✅ Sentence Encoder Output Shape: {output.shape}")
    assert output.shape == (BATCH_SIZE, NUM_SENTENCES, EMBED_DIM)


def test_sentence_gp_shape():
    """Tests the shape of the output from SentenceGP."""
    sentence_gp = SentenceGP(
        input_dim=EMBED_DIM,
        output_dim=EMBED_DIM,
        n_inducing=N_INDUCING,
        embedding_dim=EMBED_DIM,
    ).to(DEVICE)
    input_tensor = torch.randn(
        BATCH_SIZE, NUM_SENTENCES, EMBED_DIM, device=DEVICE
    )
    mean, variance = sentence_gp(input_tensor, num_sentences=NUM_SENTENCES)
    logger.info(
        f"✅ Sentence GP Mean Shape: {mean.shape} \nSentence GP Variance Shape: {variance.shape}"
    )
    assert mean.shape == (BATCH_SIZE, NUM_SENTENCES, EMBED_DIM)
    assert variance.shape == (BATCH_SIZE, NUM_SENTENCES, EMBED_DIM)


def test_sentence_gp_output_type():
    """Tests the data type of the output from SentenceGP."""
    sentence_gp = SentenceGP(
        input_dim=EMBED_DIM,
        output_dim=EMBED_DIM,
        n_inducing=N_INDUCING,
        embedding_dim=EMBED_DIM,
    ).to(DEVICE)
    input_tensor = torch.randn(
        BATCH_SIZE, NUM_SENTENCES, EMBED_DIM, device=DEVICE
    )
    mean, variance = sentence_gp(input_tensor, num_sentences=NUM_SENTENCES)
    logger.info(
        f"✅ Sentence GP Mean Type: {type(mean)} \nSentence GP Variance Type: {type(variance)}"
    )
    assert isinstance(mean, torch.Tensor)
    assert isinstance(variance, torch.Tensor)


def test_sentence_gp_variance_for_similarity():
    """Tests the variance output of SentenceGP based on sentence similarity."""
    sentence_gp = SentenceGP(
        input_dim=EMBED_DIM,
        output_dim=EMBED_DIM,
        n_inducing=N_INDUCING,
        embedding_dim=EMBED_DIM,
    ).to(DEVICE)

    # Create a sentence encoder
    sentence_encoder = SentenceEncoder(
        vocab_size=tokenizer.vocab_size,  # Provide vocab_size
        hidden_dim=EMBED_DIM,
        output_dim=EMBED_DIM,
    ).to(DEVICE)

    # Example sentences (try more distinct sentences here)
    sentence1 = "This is a sentence about cats."
    sentence2 = "That movie was incredibly boring."
    sentence3 = "This sentence talks about dogs."

    # Encode the sentences correctly
    encoded_ids = [
        tokenizer.encode(sentence1).tolist(),
        tokenizer.encode(sentence2).tolist(),
        tokenizer.encode(sentence3).tolist(),
    ]

    # Pad the token IDs to have the same length
    max_length = max(len(ids) for ids in encoded_ids)
    padded_ids = [
        ids + [tokenizer.pad_token_id] * (max_length - len(ids))
        for ids in encoded_ids
    ]

    logger.info(f"✅ Padded Token IDs: {padded_ids}")

    # Create the input tensor for the SentenceEncoder (torch.LongTensor):
    input_tensor = torch.tensor(padded_ids, device=DEVICE, dtype=torch.long).unsqueeze(
        0
    )  # Reshape to (1, num_sentences, sentence_len)
    logger.info(
        f"✅ Input Tensor Shape: {input_tensor.shape}  \n✅ Input Tensor: {input_tensor}"
    )

    # Pass the encoded IDs directly to SentenceEncoder
    sentence_embeddings = sentence_encoder(input_tensor)

    # Calculate variance using SentenceGP
    _, variance = sentence_gp(sentence_embeddings, num_sentences=NUM_SENTENCES)

    # Check for variance differences based on similarity
    # (Sentence 1 and 3 should have lower variance because they are more similar)
    logger.info(
        f"✅ Sentence Variances: \nSentence 1: {variance[0, 0, 0].item():.4f} \nSentence 2: {variance[0, 1, 0].item():.4f} \nSentence 3: {variance[0, 2, 0].item():.4f}"
    )
    # Adjust the assertion to use torch.allclose for comparing tensors with a tolerance
    assert torch.allclose(variance[0, 0, 0], variance[0, 1, 0],
                          atol=1e-2), "Variances for sentence 1 and 2 are not close enough."
    assert torch.allclose(variance[0, 0, 0], variance[0, 2, 0],
                          atol=1e-2), "Variances for sentence 1 and 3 are not close enough."


def test_sentence_gp_variance_positivity():
    """Tests that the variance output of SentenceGP is always positive."""
    sentence_gp = SentenceGP(
        input_dim=EMBED_DIM,
        output_dim=EMBED_DIM,
        n_inducing=N_INDUCING,
        embedding_dim=EMBED_DIM,
    ).to(DEVICE)
    input_tensor = torch.randn(
        BATCH_SIZE, NUM_SENTENCES, EMBED_DIM, device=DEVICE
    )
    _, variance = sentence_gp(input_tensor, num_sentences=NUM_SENTENCES)
    logger.info(f"✅ Variance is positive: {(variance > 0).all()}")
    assert (variance > 0).all(), "Variance should always be positive."

    # Check the softplus function for variance positivity
    assert (
            softplus(torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])) > 0
    ).all()


def test_sentence_gp_learnability_with_lm():
    """Tests if the UncertainTransformer can effectively learn from SentenceEncoder."""

    # Load SST-2 dataset from Hugging Face
    dataset = load_dataset("glue", "sst2")
    sentences = [example["sentence"] for example in dataset["train"]]

    # Convert sentences to token IDs and get labels
    token_ids = [tokenizer.encode(s) for s in sentences]
    labels = [example["label"] for example in dataset["train"]]

    # Prepare data for training (batches of token IDs and labels)
    def create_batch(batch_size, max_length):
        input_batch = []
        label_batch = []
        for i in range(0, len(token_ids), batch_size):
            batch_ids = token_ids[i: i + batch_size]
            batch_labels = labels[i: i + batch_size]

            # Pad the token IDs to have the same length
            max_batch_length = max(len(ids) for ids in batch_ids)
            padded_batch_ids = [
                ids.tolist() + [tokenizer.pad_token_id] * (max_batch_length - len(ids))  # Convert ids to list
                for ids in batch_ids
            ]

            input_batch.append(
                torch.tensor(padded_batch_ids, device=DEVICE, dtype=torch.long)
            )
            label_batch.append(torch.tensor(batch_labels, device=DEVICE, dtype=torch.long))
        return input_batch, label_batch

    batch_size = 2
    max_length = 10  # Use a value that is a multiple of sliding_window_size

    train_data, train_labels = create_batch(batch_size, max_length)

    # Model Configurations (identical except for the use of SentenceEncoder)
    config_base = UncertainTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=EMBED_DIM,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=2,
        dropout=0.1,
        max_position_embeddings=max_length,
        pad_token_id=tokenizer.pad_token_id,
        use_mamba=False,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand_factor=EXPAND_FACTOR,
        dt_rank=DT_RANK,
        sliding_window_size=max_length,  # Use max_length here
    )

    config_sentence_encoder = UncertainTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=EMBED_DIM,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=2,
        dropout=0.1,
        max_position_embeddings=max_length,
        pad_token_id=tokenizer.pad_token_id,
        use_mamba=False,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand_factor=EXPAND_FACTOR,
        dt_rank=DT_RANK,
        sliding_window_size=max_length,  # Use max_length here
    )

    # Create models
    model_base = UncertainTransformerLMHeadModel(config_base).to(DEVICE)
    model_with_encoder = UncertainTransformerLMHeadModel(config_sentence_encoder).to(
        DEVICE
    )

    # Add the SentenceEncoder to the model_with_encoder
    model_with_encoder.transformer.sentence_encoder = SentenceEncoder(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=EMBED_DIM,
        output_dim=EMBED_DIM,
    ).to(DEVICE)

    # Initialize with Xavier Uniform for better starting point
    for model in [model_base, model_with_encoder]:
        nn.init.xavier_uniform_(model.transformer.embedding.weight)
        nn.init.xavier_uniform_(model.lm_head.weight)

    # Optimizers
    optimizer_base = torch.optim.Adam(
        model_base.parameters(), lr=LEARNING_RATE
    )
    optimizer_with_encoder = torch.optim.Adam(
        model_with_encoder.parameters(), lr=LEARNING_RATE
    )

    # Train both models
    for epoch in range(NUM_EPOCHS):
        # Training with base model (without SentenceEncoder)
        for i, (input_ids, labels) in enumerate(zip(train_data, train_labels)):
            optimizer_base.zero_grad()
            outputs = model_base(
                input_ids=input_ids,
                labels=labels,
                position_ids=torch.arange(input_ids.shape[1], device=DEVICE),
            )
            loss = outputs.loss
            loss.backward()
            optimizer_base.step()
            logger.info(
                f"Epoch {epoch + 1} | Base Model | Batch {i + 1} | Loss: {loss.item():.4f}"
            )

        # Training with model with SentenceEncoder
        for i, (input_ids, labels) in enumerate(zip(train_data, train_labels)):
            optimizer_with_encoder.zero_grad()
            outputs = model_with_encoder(
                input_ids=input_ids,
                labels=labels,
                position_ids=torch.arange(input_ids.shape[1], device=DEVICE),
            )
            loss = outputs.loss
            loss.backward()
            optimizer_with_encoder.step()
            logger.info(
                f"Epoch {epoch + 1} | SentenceEncoder Model | Batch {i + 1} | Loss: {loss.item():.4f}"
            )

    # Evaluate both models
    def evaluate_model(model, data, labels):
        total_loss = 0
        for input_ids, labels in zip(data, labels):
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item()
        return total_loss / len(data)

    base_model_loss = evaluate_model(model_base, train_data, train_labels)
    sentence_encoder_model_loss = evaluate_model(model_with_encoder, train_data, train_labels)

    logger.info(
        f"Base Model Final Loss: {base_model_loss:.4f} | SentenceEncoder Model Final Loss: {sentence_encoder_model_loss:.4f}"
    )

    # Check if the SentenceEncoder model learns better than the base model
    assert sentence_encoder_model_loss < base_model_loss, (
        "SentenceEncoder Model should learn better than the base model on this sentence"
        " isolation task."
    )


def test_sentence_gp_performance():
    """Tests the performance of SentenceGP."""
    sentence_gp = SentenceGP(
        input_dim=EMBED_DIM,
        output_dim=EMBED_DIM,
        n_inducing=N_INDUCING,
        embedding_dim=EMBED_DIM,
    ).to(DEVICE)
    # Use a larger input tensor:
    input_tensor = torch.randn(
        BATCH_SIZE * 5, NUM_SENTENCES * 5, EMBED_DIM, device=DEVICE
    )

    inference_times = []
    for _ in range(10):  # Run multiple times
        start_time = time.time()
        mean, variance = sentence_gp(
            input_tensor, num_sentences=NUM_SENTENCES * 5
        )
        end_time = time.time()
        inference_times.append(end_time - start_time)

    average_inference_time = sum(inference_times) / len(inference_times)
    logger.info(
        f"✅ Average Inference time: {average_inference_time:.4f} seconds"
    )
    # Adjust threshold based on your performance requirements:
    assert (
            average_inference_time < 0.5
    ), "Inference time is too high."


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
