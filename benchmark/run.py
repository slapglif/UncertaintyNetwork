# run.py

import mlflow
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, PreTrainedModel

from benchmark.engine.config import BenchmarkConfig
from benchmark.engine.data_utils import get_dataset_loader
from benchmark.engine.main import run_comprehensive_benchmark
from core.models.uncertainty.uncertainty import UncertainTransformerLMHeadModel, UncertainTransformerConfig


def main():
    mlflow.start_run()

    # Load WikiText-2 dataset
    wikitext_config = BenchmarkConfig(data_path="wikitext-2-v1", max_length=512, batch_size=32)
    wikitext_dataloader = get_dataset_loader("misc", wikitext_config, cache_dir=".cache")

    # Define model configurations
    model_configs = [
        UncertainTransformerConfig(n_layers=2, d_model=128, n_heads=2),
        UncertainTransformerConfig(n_layers=4, d_model=256, n_heads=4),
        UncertainTransformerConfig(n_layers=6, d_model=512, n_heads=8),
    ]

    # Pretrain and benchmark models
    for i, config in enumerate(model_configs):
        model = UncertainTransformerLMHeadModel(config)

        # Pretrain on WikiText-2
        pretrain(model, wikitext_dataloader)

        # Benchmark on various datasets
        datasets = ["qa", "coding", "generalization", "medical", "misc"]
        for dataset in datasets:
            data_config = BenchmarkConfig(data_path=dataset, max_length=512, batch_size=32)
            data_module = get_dataset_loader(dataset, data_config, cache_dir=".cache")

            results = run_comprehensive_benchmark(model, model.tokenizer, data_module, data_config)

            # Log results to MLflow
            for task, task_results in results.items():
                for metric, value in task_results.items():
                    mlflow.log_metric(f"model_{i}_{dataset}_{task}_{metric}", value)

        # Save the model
        torch.save(model.state_dict(), f"model_{i}.pt")
        mlflow.log_artifact(f"model_{i}.pt")

    mlflow.end_run()


def pretrain(model: PreTrainedModel, dataloader: DataLoader, num_epochs=3):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(dataloader) * num_epochs)

    model.train()
    for _ in range(num_epochs):
        for batch in dataloader:
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            mlflow.log_metric("pretrain_loss", loss.item())


if __name__ == "__main__":
    main()
