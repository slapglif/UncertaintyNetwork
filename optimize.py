import optuna
import pytorch_lightning as pl
import torch
from loguru import logger
from transformers import GPT2Tokenizer, get_cosine_schedule_with_warmup

from core.data.datamodule import SlimPajamaDataModule
from core.models.layers import MambaLayer, TransformerEncoderLayer
from core.models.uncertain_nn import UncertainTransformerConfig, UncertainTransformerLMHeadModel
from core.tests.test_generation import calculate_perplexity
torch.set_float32_matmul_precision('high')

def objective(trial: optuna.Trial):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.Trial): An Optuna trial object.

    Returns:
        float: The perplexity on the validation set.
    """

    # Mamba Layer Hyperparameters
    d_state = trial.suggest_int("d_state", 4, 64, step=4)  # State dimension
    d_conv = trial.suggest_int("d_conv", 2, 8, step=2)  # Convolution kernel size
    expand_factor = trial.suggest_float("expand_factor", 1.5, 3.0, step=0.5)  # Expansion factor
    dt_rank = trial.suggest_int("dt_rank", 4, 32, step=4)  # Rank of dt parameter
    dt_min = trial.suggest_float("dt_min", 1e-4, 1e-2, log=True)  # Minimum value for dt
    dt_max = trial.suggest_float("dt_max", 1e-2, 1.0, log=True)  # Maximum value for dt

    # Clipping Ranges (Experiment with these - using linear scale now)
    exp_term_clip_min = trial.suggest_float("exp_term_clip_min", 1e-6, 1e-3)
    exp_term_clip_max = trial.suggest_float("exp_term_clip_max", 1e3, 1e6)
    y_clip_min = trial.suggest_float("y_clip_min", -1e5, -1e2)
    y_clip_max = trial.suggest_float("y_clip_max", 1e2, 1e5)

    # Create Mamba Layer with suggested hyperparameters
    config = UncertainTransformerConfig(
        vocab_size=50257,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_layers=6,
        dropout=0.1,
        max_position_embeddings=1024,
        pad_token_id=50256,
        d_state=d_state,
        d_conv=d_conv,
        mamba_expand_factor=expand_factor,
        mamba_dt_rank=dt_rank,
        mamba_dt_min=dt_min,
        mamba_dt_max=dt_max,
    )
    mamba_layer = MambaLayer(config)

    class SAMBAModel(UncertainTransformerLMHeadModel, pl.LightningModule):  # Inherit from pl.LightningModule
        def __init__(self, config):
            super().__init__(config)
            self.transformer.layers = torch.nn.ModuleList(
                [
                    # Assuming you're interleaving Mamba with other layers
                    # Replace this with your actual layer arrangement
                    TransformerEncoderLayer(config) if i % 2 == 0 else mamba_layer
                    for i in range(config.n_layers)
                ]
            )

        # Add necessary LightningModule methods here (training_step, configure_optimizers, etc.)
        def training_step(self, batch, batch_idx):
            input_ids, labels = batch
            attention_mask = (input_ids != self.config.pad_token_id).long()
            outputs = self.forward(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss

        def validation_step(self, batch, batch_idx):
            input_ids, labels = batch
            attention_mask = (input_ids != self.hparams["pad_token_id"]).long()
            outputs = self.forward(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            self.log(
                "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
            )

            perplexity = self.perplexity(
                outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1)
            )
            self.log(
                "val_perplexity",
                perplexity,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

            if batch_idx == 0:
                input_ids = batch[0][:1]
                generated = self.model.generate(input_ids, max_length=50)
                generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                self.logger.experiment.add_text("generated_text", generated_text, self.current_epoch)

            return loss

        def configure_optimizers(self):
            # Prepare optimizer
            param_optimizer = list(self.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in param_optimizer if all(nd not in n for nd in no_decay)
                    ],
                    "weight_decay": self.hparams["weight_decay"],
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams["learning_rate"],
                betas=(0.9, 0.999),
                eps=1e-8,
            )

            # Prepare scheduler
            num_training_steps = self.trainer.estimated_stepping_batches
            num_warmup_steps = int(num_training_steps * self.hparams["warmup_ratio"])
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

    model = SAMBAModel(config)

    # Train and Evaluate Your SAMBA Model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    datamodule = SlimPajamaDataModule(tokenizer=tokenizer,
                                      max_length=config.max_position_embeddings,
                                      batch_size=32)  # Adjust batch size as needed
    trainer = pl.Trainer(max_epochs=5,  # Adjust as needed
                         accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.fit(model, datamodule=datamodule)

    # Calculate Perplexity on Validation Set
    val_perplexity = calculate_perplexity(model, tokenizer, datamodule.val_dataset)

    logger.info(f"Trial {trial.number}: Validation Perplexity = {val_perplexity:.4f}")
    logger.info(f"Hyperparameters: {trial.params}")

    return val_perplexity


# Optuna Study
study = optuna.create_study(direction="minimize")  # Minimize perplexity
study.optimize(objective, n_trials=100)  # Adjust the number of trials

# Best Hyperparameters
best_params = study.best_params
logger.info(f"Best Hyperparameters: {best_params}")
