# benchmark_utils.py

from typing import Dict, Any, List

import torch
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from benchmark.engine.config import BenchmarkConfig
from core.utils.metrics import calculate_bleu_score, calculate_rouge_scores
from benchmark.engine.uncertainty_utils import expected_calibration_error, uncertainty_decomposition, out_of_distribution_detection


def run_benchmark(model, tokenizer, data_module, config: BenchmarkConfig) -> Dict[str, Any]:
    results = {}

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        for task in config.benchmark_tasks:
            task_id = progress.add_task(f"Running {task}", total=100)
            logger.info(f"Starting benchmark for task: {task}")

            if task == "language_modeling":
                results[task] = benchmark_language_modeling(model, data_module.test_dataloader(), progress, task_id)
            elif task == "text_generation":
                results[task] = benchmark_text_generation(model, tokenizer, data_module.test_dataloader(), config,
                                                          progress, task_id)
            elif task == "uncertainty_estimation":
                results[task] = benchmark_uncertainty_estimation(model, data_module.test_dataloader(), config, progress,
                                                                 task_id)

            progress.update(task_id, completed=100)

    return results


def benchmark_language_modeling(model, dataloader, progress, task_id) -> Dict[str, float]:
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_loss += loss.item() * input_ids.size(0)
            total_tokens += attention_mask.sum().item()

            progress.update(task_id, advance=100 / len(dataloader))

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return {"perplexity": perplexity}


def benchmark_text_generation(model, tokenizer, dataloader, config: BenchmarkConfig, progress, task_id) -> Dict[
    str, float]:
    model.eval()
    generated_texts = []
    reference_texts = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        generated_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=config.max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        generated_texts.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
        reference_texts.extend(tokenizer.batch_decode(batch["labels"], skip_special_tokens=True))

        progress.update(task_id, advance=100 / len(dataloader))

    bleu_score = calculate_bleu_score(reference_texts, generated_texts)
    rouge_scores = calculate_rouge_scores(reference_texts, generated_texts)

    return {
        "bleu": bleu_score,
        "rouge-1": rouge_scores["rouge-1"],
        "rouge-2": rouge_scores["rouge-2"],
        "rouge-l": rouge_scores["rouge-l"]
    }


def benchmark_uncertainty_estimation(model, dataloader, config: BenchmarkConfig, progress, task_id) -> Dict[str, float]:
    model.eval()
    uncertainties = []
    accuracies = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs, uncertainty = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            predictions = logits.argmax(dim=-1)
            accuracies.extend((predictions == labels).float().cpu().numpy())
            uncertainties.extend(uncertainty.mean(dim=-1).cpu().numpy())

            progress.update(task_id, advance=100 / len(dataloader))

    ece = expected_calibration_error(torch.tensor(uncertainties), torch.tensor(accuracies))
    aleatoric, epistemic = uncertainty_decomposition(torch.tensor(uncertainties))

    # Generate out-of-distribution data
    ood_dataloader = generate_ood_data(config)
    ood_uncertainties = []

    with torch.no_grad():
        for batch in ood_dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            _, uncertainty = model(input_ids, attention_mask=attention_mask)
            ood_uncertainties.extend(uncertainty.mean(dim=-1).cpu().numpy())

    auroc, auprc = out_of_distribution_detection(torch.tensor(uncertainties), torch.tensor(ood_uncertainties))

    return {
        "expected_calibration_error": ece,
        "aleatoric_uncertainty": aleatoric.mean().item(),
        "epistemic_uncertainty": epistemic.mean().item(),
        "ood_auroc": auroc,
        "ood_auprc": auprc
    }


def generate_ood_data(config: BenchmarkConfig):
    # This is a placeholder. In a real scenario, you'd generate or load OOD data.
    # For now, we'll just return a small random dataset
    class RandomDataset(torch.utils.data.Dataset):
        def __init__(self, size=1000, seq_len=128, vocab_size=50257):
            self.data = torch.randint(0, vocab_size, (size, seq_len))
            self.attention_mask = torch.ones_like(self.data)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return {
                "input_ids": self.data[idx],
                "attention_mask": self.attention_mask[idx]
            }

    dataset = RandomDataset()
    return torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)


def compute_metrics(train_losses: List[float], val_losses: List[float]) -> Dict[str, float]:
    return {
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "best_val_loss": min(val_losses),
        "train_loss_reduction": train_losses[0] - train_losses[-1],
        "val_loss_reduction": val_losses[0] - val_losses[-1],
    }
