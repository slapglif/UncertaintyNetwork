#!/usr/bin/env python3
# main.py

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()

import pytorch_lightning as pl
import torch
from lightning import LightningModule
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from rich import print as rprint
from torch.utils.data import DataLoader

from benchmark.engine.augmentation import apply_augmentations
from benchmark.engine.metrics import calculate_uncertainty_metrics, calculate_auc_roc, calculate_log_loss, \
    calculate_mutual_information, calculate_predictive_entropy, calculate_diversity_metrics, calculate_bleu_score, \
    calculate_rouge_scores, calculate_perplexity, calculate_accuracy, calculate_precision_recall_f1
from benchmark.engine.config import BenchmarkConfig
from benchmark.engine.data_utils import get_dataset_loader

# Configure loguru for advanced logging
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])


def setup_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="🚀 Advanced Benchmarking Tool for Uncertain Transformers")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--mode", choices=["train", "finetune", "benchmark"], default="benchmark",
                        help="Operation mode")
    parser.add_argument("--output", type=str, default="results", help="Output directory for results and visualizations")
    parser.add_argument("--dataset", choices=["qa", "coding", "generalization", "medical", "misc"], default="misc",
                        help="Dataset to use")
    parser.add_argument("--cache_dir", type=str, default=".cache", help="Directory to cache datasets")
    return parser


def setup_environment(config: BenchmarkConfig):
    pl.seed_everything(config.seed)
    torch.set_float32_matmul_precision('high')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_trainer(config: BenchmarkConfig) -> pl.Trainer:
    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3),
        EarlyStopping(monitor="val_loss", patience=config.early_stopping_patience, mode="min"),
        LearningRateMonitor(logging_interval="step")
    ]

    logger = TensorBoardLogger("logs", name="uncertain_transformer")

    return pl.Trainer(
        max_epochs=config.max_epochs,
        callbacks=callbacks,
        logger=logger,
        gpus=torch.cuda.device_count(),
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
        precision=16 if config.use_mixed_precision else 32,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
        progress_bar_refresh_rate=0,  # Disable default progress bar
    )


def run_training(model: pl.LightningModule, data_module: pl.LightningDataModule, trainer: pl.Trainer):
    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Training", total=trainer.max_epochs)
        for epoch in range(trainer.max_epochs):
            trainer.fit(model, data_module)
            progress.update(task, advance=1)


def display_results(results: Dict[str, Any]):
    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    for category, metrics in results.items():
        table.add_row(category.capitalize(), "")
        for metric, value in metrics.items():
            table.add_row(f"  {metric}", f"{value:.4f}" if isinstance(value, float) else str(value))

    console.print(table)


def save_results(results: Dict[str, Any], output_dir: str):
    output_path = Path(output_dir) / "results.json"
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    console.print(f"Results saved to: [bold]{output_path}[/bold]")


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    rprint(Panel.fit("🚀 [bold cyan]Advanced Benchmarking Tool for Uncertain Transformers[/bold cyan] 🚀"))

    config = BenchmarkConfig.from_json(args.config)
    setup_environment(config)

    console.print(f"⚙️  Loaded configuration from: [bold]{args.config}[/bold]")

    # Set up data module with caching
    data_module = get_dataset_loader(args.dataset, config, cache_dir=args.cache_dir)


class AdvancedUncertainTransformerLightningModule(LightningModule):
    pass


def setup_training(config: BenchmarkConfig, model: LightningModule) -> pl.Trainer:
    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3),
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        LearningRateMonitor(logging_interval="step")
    ]

    logger = TensorBoardLogger("logs", name="uncertain_transformer")

    console.print("[bold green]✨ All operations completed successfully! ✨[/bold green]")

    return pl.Trainer(
        max_epochs=config.max_epochs,
        callbacks=callbacks,
        logger=logger,
        gpus=torch.cuda.device_count(),
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
        precision=16 if config.use_mixed_precision else 32,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
    )


# misc.py (add this to the existing file)


def run_comprehensive_benchmark(model, tokenizer, data_module, config):
    results = {}
    tasks = [
        ("Language Modeling", benchmark_language_modeling),
        ("Text Generation", benchmark_text_generation),
        ("Uncertainty Estimation", benchmark_uncertainty_estimation),
        ("Augmentation Impact", benchmark_augmentation_impact),
        ("Diversity", benchmark_diversity)
    ]

    console.print(Panel.fit("🚀 [bold cyan]Starting Comprehensive Benchmark[/bold cyan] 🚀"))

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
    ) as progress:
        overall_task = progress.add_task("[yellow]Overall Progress", total=len(tasks))

        for task_name, task_func in tasks:
            task_progress = progress.add_task(f"[cyan]{task_name}", total=100)
            try:
                console.print(f"\n[bold green]Starting {task_name}...[/bold green]")
                start_time = time.time()

                # Run the benchmark task
                task_results = task_func(model, tokenizer, data_module, config, progress, task_progress)

                end_time = time.time()
                duration = end_time - start_time

                results[task_name] = task_results
                console.print(f"[bold green]{task_name} completed in {duration:.2f} seconds.[/bold green]")

                # Display task-specific results
                display_task_results(task_name, task_results)

            except Exception as e:
                console.print(f"[bold red]Error in {task_name}:[/bold red]")
                console.print(Panel(str(e), title="Error Details", border_style="red"))
                console.print(traceback.format_exc())
                console.print(f"[yellow]Skipping {task_name} due to error. Continuing with next task...[/yellow]")
                results[task_name] = {"error": str(e)}

            progress.update(overall_task, advance=1)
            progress.update(task_progress, completed=100)

    console.print(Panel.fit("[bold green]Comprehensive Benchmark Completed![/bold green]"))
    return results


def display_task_results(task_name, results):
    table = Table(title=f"{task_name} Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    for metric, value in results.items():
        if isinstance(value, float):
            table.add_row(metric, f"{value:.4f}")
        else:
            table.add_row(metric, str(value))

    console.print(table)


def benchmark_language_modeling(model, tokenizer, data_module, config, progress, task_id):
    console.print("[cyan]Evaluating language modeling performance...[/cyan]")
    model.eval()
    total_loss = 0
    total_tokens = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_module.test_dataloader():
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item() * input_ids.size(0)
            total_tokens += attention_mask.sum().item()
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

            progress.update(task_id, advance=100 / len(data_module.test_dataloader()))

    perplexity = calculate_perplexity(total_loss / total_tokens)
    accuracy = calculate_accuracy(all_preds, all_labels)
    precision_recall_f1 = calculate_precision_recall_f1(all_preds, all_labels)

    return {
        "perplexity": perplexity,
        "accuracy": accuracy,
        **precision_recall_f1
    }


def benchmark_text_generation(model, tokenizer, data_module, config, progress, task_id):
    console.print("[cyan]Evaluating text generation capabilities...[/cyan]")
    model.eval()
    generated_texts = []
    reference_texts = []

    for batch in data_module.test_dataloader():
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

        progress.update(task_id, advance=100 / len(data_module.test_dataloader()))

    bleu_score = calculate_bleu_score(reference_texts, generated_texts)
    rouge_scores = calculate_rouge_scores(reference_texts, generated_texts)
    diversity_metrics = calculate_diversity_metrics(generated_texts)

    return {
        "bleu": bleu_score,
        **rouge_scores,
        **diversity_metrics
    }


def benchmark_uncertainty_estimation(model, tokenizer, data_module, config, progress, task_id):
    console.print("[cyan]Evaluating uncertainty estimation...[/cyan]")
    model.eval()
    uncertainties = []
    accuracies = []
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in data_module.test_dataloader():
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs, uncertainty = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            predictions = logits.argmax(dim=-1)
            accuracies.extend((predictions == labels).float().cpu().numpy())
            uncertainties.extend(uncertainty.mean(dim=-1).cpu().numpy())
            all_logits.append(logits.cpu())
            all_labels.extend(labels.cpu().numpy())

            progress.update(task_id, advance=100 / len(data_module.test_dataloader()))

    all_logits = torch.cat(all_logits, dim=0)
    uncertainty_metrics = calculate_uncertainty_metrics(uncertainties, accuracies)
    auc_roc = calculate_auc_roc(uncertainties, accuracies)
    log_loss = calculate_log_loss(torch.softmax(all_logits, dim=-1).numpy(), all_labels)
    mutual_info = calculate_mutual_information(all_logits)
    predictive_entropy = calculate_predictive_entropy(all_logits)

    return {
        **uncertainty_metrics,
        "auc_roc": auc_roc,
        "log_loss": log_loss,
        "mutual_information": mutual_info,
        "predictive_entropy": predictive_entropy
    }


def benchmark_augmentation_impact(model, tokenizer, data_module, config, progress, task_id):
    console.print("[cyan]Evaluating impact of data augmentation...[/cyan]")
    original_perplexity = benchmark_language_modeling(model, tokenizer, data_module, config, progress, task_id)[
        "perplexity"]

    # Apply augmentation to the test dataset
    augmented_dataset = apply_augmentations(data_module.test_dataset, config)
    augmented_dataloader = DataLoader(augmented_dataset, batch_size=config.batch_size, num_workers=config.num_workers)

    augmented_perplexity = \
        benchmark_language_modeling(model, tokenizer, augmented_dataloader, config, progress, task_id)["perplexity"]

    return {
        "original_perplexity": original_perplexity,
        "augmented_perplexity": augmented_perplexity,
        "perplexity_change": augmented_perplexity - original_perplexity
    }


def benchmark_diversity(model, tokenizer, data_module, config, progress, task_id):
    console.print("[cyan]Evaluating output diversity...[/cyan]")
    model.eval()
    generated_texts = []

    with torch.no_grad():
        for batch in data_module.test_dataloader():
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            generated_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=config.max_length,
                num_return_sequences=5,  # Generate multiple sequences for diversity
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )

            generated_texts.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

            progress.update(task_id, advance=100 / len(data_module.test_dataloader()))

    return calculate_diversity_metrics(generated_texts)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred during execution:")
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)
