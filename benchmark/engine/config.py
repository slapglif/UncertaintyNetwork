# config.py

from dataclasses import dataclass, field
from typing import List, Optional
import json

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 3072
    n_layers: int = 12
    dropout: float = 0.1
    max_position_embeddings: int = 1024
    pad_token_id: int = 0
    use_mamba: bool = True
    d_state: int = 16
    d_conv: int = 4
    expand_factor: float = 2.0
    dt_rank: int = 8
    n_inducing: int = 5

@dataclass
class BenchmarkConfig:
    # General configuration
    seed: int = 42
    output_dir: str = "benchmark_results"

    # Data configuration
    data_path: str = field(default="data/slimpajama")
    tokenizer_path: str = field(default="tokenizer")
    max_length: int = 1024
    batch_size: int = 32
    num_workers: int = 4

    # Model configuration
    model_params: ModelConfig = field(default_factory=ModelConfig)
    use_mamba: bool = True
    uncertainty_weight: float = 0.1

    # Training configuration
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 10
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    use_mixed_precision: bool = True
    early_stopping_patience: int = 3

    # Uncertainty configuration
    n_gp_layers: int = 1
    mc_samples: int = 10
    calibration_method: str = "temperature_scaling"

    # Benchmarking configuration
    benchmark_tasks: List[str] = field(default_factory=lambda: ["language_modeling", "text_generation", "uncertainty_estimation"])
    metrics: List[str] = field(default_factory=lambda: ["perplexity", "bleu", "rouge", "uncertainty"])

    # Knowledge Distillation
    use_knowledge_distillation: bool = False
    teacher_model_path: Optional[str] = None

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_json(self, json_path: str):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)