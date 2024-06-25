from sys import stdout
from typing import Dict, List

import torch
from datasets import load_dataset
from loguru import logger
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util

from core.models.uncertainty.uncertain_nn import (
    UncertainTransformerConfig,
    UncertainTransformerLMHeadModel,
)
from core.utils.tokenizer import Tokenizer
from core.utils.utils import generate_text

# Configure Loguru
logger.remove()  # Remove default logger
logger.add(
    stdout,  # Log file
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)

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
model.to("cuda")
model.eval()


def evaluate_qa(
    model: UncertainTransformerLMHeadModel,
    tokenizer: Tokenizer,
    qa_dataset: Dict,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    """
    Evaluates the model's zero-shot question answering ability on a QA corpus using RAG.

    Args:
        model (UncertainTransformerLMHeadModel): The model to evaluate.
        tokenizer (Tokenizer): The tokenizer to use.
        qa_dataset (Dict): The BoolQ dataset.
        device (torch.device, optional): The device to run the model on. Defaults to CUDA if available, else CPU.
    """
    logger.info("🚀 Starting QA Evaluation! 🚀")
    model.to(device)
    model.eval()

    rouge = Rouge()
    sentence_model = SentenceTransformer(
        "all-mpnet-base-v2"
    )  # Sentence embedding model
    logger.info("🤖 Loaded Sentence Embedding Model: all-mpnet-base-v2 🤖")

    total_bleu = 0
    total_rouge_l = 0

    for i, example in enumerate(qa_dataset["validation"]):
        logger.info(f"🧠 Processing QA Pair {i + 1} 🧠")
        question = example["question"]
        answer = "yes" if example["answer"] else "no"
        logger.info(f"❓ Question: {question}")
        logger.info(f"✅ Gold Answer: {answer}")

        # 1. Retrieve Relevant QA Pairs
        logger.info("🔎 Retrieving Relevant QA Pairs from Corpus 🔎")
        retrieved_pairs = retrieve_relevant_qa(
            question, qa_dataset["train"], sentence_model, top_k=3
        )
        logger.info(f"🧲 Found {len(retrieved_pairs)} Relevant Pairs 🧲")

        # 2. Construct Prompt with Retrieved Information
        logger.info("📝 Constructing Prompt with Retrieved Information 📝")
        prompt = f"Q: {question}\n"
        for pair in retrieved_pairs:
            prompt += f"Context: Q: {pair['question']}\nA: {pair['answer']}\n"
        prompt += "A:"
        logger.info(f"➡️ Prompt: {prompt}")

        # Generate text using the prompt
        logger.info("✍️ Generating Answer using the Model ✍️")
        generated_text = generate_text(
            model,
            tokenizer.tokenizer,
            prompt,
            max_length=100,  # Adjust max length as needed
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            num_return_sequences=1,
            device=device,
        )[
            0
        ]  # Get the first generated sequence
        logger.info(f"💬 Generated Answer: {generated_text}")

        # Calculate BLEU score
        logger.info("🧮 Calculating BLEU Score 🧮")
        bleu_score = sentence_bleu(
            [tokenizer.tokenize(answer)], tokenizer.tokenize(generated_text)
        )
        total_bleu += bleu_score
        logger.info(f"🔵 BLEU Score: {bleu_score:.4f}")

        # Calculate ROUGE-L score
        logger.info("🧮 Calculating ROUGE-L Score 🧮")
        rouge_scores = rouge.get_scores(generated_text, answer)
        rouge_l_score = rouge_scores[0]["rouge-l"]["f"]
        total_rouge_l += rouge_l_score
        logger.info(f"🔴 ROUGE-L Score: {rouge_l_score:.4f}")

    average_bleu = total_bleu / len(qa_dataset["validation"])
    average_rouge_l = total_rouge_l / len(qa_dataset["validation"])
    logger.info(f"\n🌟 Average BLEU Score: {average_bleu:.4f} 🌟")
    logger.info(f"🌟 Average ROUGE-L Score: {average_rouge_l:.4f} 🌟")
    logger.info("🎉 QA Evaluation Completed! 🎉")


def retrieve_relevant_qa(
    query: str,
    qa_corpus: List[Dict],
    sentence_model: SentenceTransformer,
    top_k: int = 3,
) -> List[Dict]:
    """
    Retrieves relevant QA pairs from a corpus based on semantic similarity.

    Args:
        query (str): The input question.
        qa_corpus (List[Dict]): The QA corpus.
        sentence_model (SentenceTransformer): The sentence embedding model.
        top_k (int, optional): The number of top QA pairs to retrieve. Defaults to 3.

    Returns:
        List[Dict]: A list of the top_k most relevant QA pairs.
    """
    query_embedding = sentence_model.encode(query, convert_to_tensor=True)

    # Ensure corpus_embeddings is always 2D even for single element
    if len(qa_corpus) == 1:
        corpus_embeddings = sentence_model.encode(
            [pair["question"] for pair in qa_corpus], convert_to_tensor=True
        ).unsqueeze(0)
    else:
        corpus_embeddings = sentence_model.encode(
            [pair["question"] for pair in qa_corpus], convert_to_tensor=True
        )

    similarities = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_k_indices = torch.topk(similarities, top_k).indices

    # Convert top_k_indices to a list of integers
    top_k_indices = top_k_indices.tolist()

    return [qa_corpus[i] for i in top_k_indices]


def boolq_dataset():
    """Loads the BoolQ dataset."""
    logger.info("📥 Loading BoolQ Dataset 📥")
    dataset = load_dataset("boolq")
    logger.info("✅ BoolQ Dataset Loaded Successfully! ✅")
    return dataset


# Test Function
tokenizer = Tokenizer.from_pretrained("gpt2")

if __name__ == "__main__":
    evaluate_qa(model, tokenizer, boolq_dataset(), torch.device("cuda"))
