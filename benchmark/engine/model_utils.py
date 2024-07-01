# model_utils.py
import torch
import torch.nn as nn
from loguru import logger
from torch.nn import functional as F
from transformers import AutoTokenizer

from benchmark.engine.config import BenchmarkConfig
from core.models.statespace import MambaConfig
from core.models.uncertainty.uncertainty import UncertainTransformerLMHeadModel, UncertainTransformerConfig
from core.models.uncertainty.uncertainty_layers import UncertaintyModule


def setup_model_and_tokenizer(config: BenchmarkConfig):
    logger.info("Setting up model and tokenizer...")

    if config.model_params.use_mamba:
        model = create_mamba_model(config)
        logger.info("Created Mamba-based model")
    else:
        model = create_transformer_model(config)
        logger.info("Created Transformer-based model")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    logger.info(f"Loaded tokenizer from {config.tokenizer_path}")

    if config.data_path:
        logger.info(f"Loading pre-trained model from {config.data_path}")
        state_dict = torch.load(config.data_path, map_location='cpu')
        model.load_state_dict(state_dict)

    uncertainty_module = setup_uncertainty_module(config)
    model.uncertainty_module = uncertainty_module
    logger.info("Added uncertainty module to the model")

    return model, tokenizer


def create_mamba_model(config: BenchmarkConfig) -> UncertainTransformerLMHeadModel:
    mamba_config = MambaConfig(
        d_model=config.model_params.d_model,
        d_state=config.model_params.d_state,
        d_conv=config.model_params.d_conv,
        expand_factor=config.model_params.expand_factor,
        dt_rank=config.model_params.dt_rank,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
    )

    model_config = UncertainTransformerConfig(
        vocab_size=config.model_params.vocab_size,
        d_model=config.model_params.d_model,
        n_heads=config.model_params.n_heads,
        d_ff=config.model_params.d_ff,
        n_layers=config.model_params.n_layers,
        dropout=config.model_params.dropout,
        max_position_embeddings=config.model_params.max_position_embeddings,
        pad_token_id=config.model_params.pad_token_id,
        use_mamba=True,
        mamba_config=mamba_config
    )

    return UncertainTransformerLMHeadModel(model_config)


def create_transformer_model(config: BenchmarkConfig) -> UncertainTransformerLMHeadModel:
    model_config = UncertainTransformerConfig(
        vocab_size=config.model_params.vocab_size,
        d_model=config.model_params.d_model,
        n_heads=config.model_params.n_heads,
        d_ff=config.model_params.d_ff,
        n_layers=config.model_params.n_layers,
        dropout=config.model_params.dropout,
        max_position_embeddings=config.model_params.max_position_embeddings,
        pad_token_id=config.model_params.pad_token_id,
        use_mamba=False
    )

    return UncertainTransformerLMHeadModel(model_config)


def setup_uncertainty_module(config: BenchmarkConfig) -> UncertaintyModule:
    return UncertaintyModule(
        input_dim=config.model_params.d_model,
        output_dim=config.model_params.vocab_size,
        n_gp_layers=config.n_gp_layers,
        n_inducing=config.model_params.n_inducing,
        dropout_rate=config.model_params.dropout,
        mc_samples=config.mc_samples
    )


def apply_knowledge_distillation(teacher_model: nn.Module, student_model: nn.Module, alpha: float = 0.5,
                                 temperature: float = 2.0):
    def knowledge_distillation_loss(student_logits, teacher_logits, labels, alpha, temperature):
        distillation_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1)
        ) * (temperature ** 2)
        student_loss = F.cross_entropy(student_logits, labels)
        return alpha * distillation_loss + (1 - alpha) * student_loss

    student_model.knowledge_distillation_loss = knowledge_distillation_loss
    student_model.teacher_model = teacher_model
    student_model.kd_alpha = alpha
    student_model.kd_temperature = temperature

    logger.info(f"Applied knowledge distillation with alpha={alpha} and temperature={temperature}")


def optimize_model_for_inference(model: UncertainTransformerLMHeadModel):
    logger.info("Optimizing model for inference...")
    model.eval()

    model = torch.jit.script(model)

    logger.info("Model optimized with TorchScript")
    return model


def analyze_model_complexity(model: UncertainTransformerLMHeadModel):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
