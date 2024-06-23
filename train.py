# train.py

import sys
from typing import Dict, Any

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger

from core.data.datamodule import SlimPajamaDataModule
from core.models.uncertain_nn import (
    UncertainTransformerLMHeadModel,
    UncertainTransformerConfig,
)
from core.utils.tokenizer import Tokenizer

# Configure Loguru
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

torch.set_float32_matmul_precision("high")


class UncertainTransformerLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training the UncertainTransformer model.
    """

    def __init__(self, hparams: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(hparams)

        # Model Configuration
        config = UncertainTransformerConfig(
            vocab_size=hparams["vocab_size"],
            d_model=hparams["d_model"],
            n_heads=hparams["n_heads"],
            d_ff=hparams["d_ff"],
            n_layers=hparams["n_layers"],
            dropout=hparams["dropout"],
            max_position_embeddings=hparams["max_length"],
            pad_token_id=hparams["pad_token_id"],
            use_stable_embedding=hparams["use_stable_embedding"],
            use_mamba=hparams["use_mamba"],
            d_state=hparams["d_state"],
            d_conv=hparams["d_conv"],
            expand_factor=hparams["expand_factor"],
            dt_rank=hparams["dt_rank"],
            dt_min=hparams["dt_min"],
            dt_max=hparams["dt_max"],
        )

        # Instantiate the Model
        self.model = UncertainTransformerLMHeadModel(config)
        self.tokenizer = None

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(
            self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        outputs = self(**batch)
        loss = outputs.loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(
            self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step."""
        outputs = self(**batch)
        loss = outputs.loss
        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )

        perplexity = torch.exp(loss)
        self.log(
            "val_perplexity",
            perplexity,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Sample Generation for Logging
        if batch_idx == 0:
            sample_input_ids = batch["input_ids"][:1]
            generated = self.model.generate(sample_input_ids, max_new_tokens=50)
            generated_text = self.tokenizer.decode(
                generated[0], skip_special_tokens=True
            )
            self.logger.experiment.add_text(
                "generated_text", generated_text, self.current_epoch
            )

        return {"val_loss": loss, "val_perplexity": perplexity}

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=self.hparams["learning_rate"] / 10,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def main():
    """Main training function."""
    logger.info("Starting main function...")

    # Hyperparameters
    hparams = {
        "vocab_size": 50257,
        "d_model": 512,
        "n_heads": 8,
        "d_ff": 2048,
        "n_layers": 6,
        "dropout": 0.1,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "max_length": 1024,
        "batch_size": 32,
        "accumulate_grad_batches": 4,
        "max_epochs": 10,
        "pad_token_id": 50256,
        "use_stable_embedding": True,
        "streaming": True,
        "use_mamba": True,
        "d_state": 16,
        "d_conv": 4,
        "expand_factor": 2.0,
        "dt_rank": None,
        "dt_min": 0.001,
        "dt_max": 0.1,
    }

    # Initialize Model
    logger.info("Initializing model...")
    model = UncertainTransformerLightningModule(hparams)

    # Load Tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = Tokenizer.from_pretrained("gpt2")
    model.tokenizer = tokenizer

    # Create Data Module
    logger.info("Initializing DataModule...")
    datamodule = SlimPajamaDataModule(
        tokenizer=tokenizer,
        max_length=hparams["max_length"],
        batch_size=hparams["batch_size"],
        streaming=hparams["streaming"],
    )

    # Callbacks
    logger.info("Setting up callbacks...")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="uncertain-transformer-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Logger
    logger.info("Setting up TensorBoard logger...")
    tb_logger = TensorBoardLogger("logs", name="uncertain-transformer")

    # Trainer
    logger.info("Setting up trainer...")
    trainer = pl.Trainer(
        max_epochs=hparams["max_epochs"],
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=tb_logger,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        accumulate_grad_batches=hparams["accumulate_grad_batches"],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=0.25,
    )

    # Start Training
    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)

    logger.info("Training completed.")


if __name__ == "__main__":
    main()
