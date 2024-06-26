# train.py

import os
import sys
from typing import Dict, Any, List, Optional

import pytorch_lightning as pl
import torch
from datasets import load_dataset, Dataset
from loguru import logger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from core.models.uncertainty.uncertain_nn import (
    UncertainTransformerLMHeadModel,
    UncertainTransformerConfig,
)
from core.models.uncertainty.uncertainty_utils import (
    uncertainty_weighted_loss,
    total_uncertainty,
)
from core.utils.tokenizer import Tokenizer

# Configure Loguru
log_level = os.environ.get("LOG_LEVEL", "INFO")
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=log_level,
)

# Enable tensor core support
torch.set_float32_matmul_precision("high")


class UncertainTransformerLightningModule(pl.LightningModule):
    def __init__(self, hparams: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(hparams)

        config = UncertainTransformerConfig(
            vocab_size=hparams["vocab_size"],
            d_model=hparams["d_model"],
            n_heads=hparams["n_heads"],
            d_ff=hparams["d_ff"],
            n_layers=hparams["n_layers"],
            dropout=hparams["dropout"],
            max_position_embeddings=hparams["max_length"],
            pad_token_id=hparams["pad_token_id"],
            use_mamba=hparams["use_mamba"],
            d_state=hparams["d_state"],
            d_conv=hparams["d_conv"],
            expand_factor=hparams["expand_factor"],
            dt_rank=hparams["dt_rank"],
            dt_min=hparams["dt_min"],
            dt_max=hparams["dt_max"],
            sliding_window_size=hparams["sliding_window_size"],
        )

        self.model = UncertainTransformerLMHeadModel(config)
        self.tokenizer = None

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(
            self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outputs, uncertainty = self(**batch)
        loss = outputs.loss

        # Apply uncertainty-weighted loss
        weighted_loss = uncertainty_weighted_loss(loss, total_uncertainty(uncertainty))

        self.log(
            "train_loss",
            weighted_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return weighted_loss

    def validation_step(
            self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        outputs, uncertainty = self(**batch)
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

        # Log mean uncertainty
        mean_uncertainty = uncertainty.mean()
        self.log(
            "val_mean_uncertainty",
            mean_uncertainty,
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

        return {
            "val_loss": loss,
            "val_perplexity": perplexity,
            "val_uncertainty": mean_uncertainty,
        }

    def configure_optimizers(self):
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


class TinyShakespeareDataModule(pl.LightningDataModule):
    def __init__(
            self,
            tokenizer: Tokenizer,
            max_length: int = 1024,
            batch_size: int = 32,
            num_workers: int = 4,
    ):
        """
        Initialize the TinyShakespeareDataModule.

        Args:
            tokenizer (Tokenizer): The tokenizer to use for processing text.
            max_length (int): Maximum sequence length for tokenization.
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of workers for DataLoaders.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """
        Set up the datasets for each stage (fit, validate, test).

        Args:
            stage (Optional[str]): The stage to set up. Can be 'fit', 'validate', or 'test'.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = TinyShakespeareDataset(
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )
            self.val_dataset = TinyShakespeareDataset(
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )

        if stage == "test" or stage is None:
            self.test_dataset = TinyShakespeareDataset(
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=self._collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
            persistent_workers=True,
        )

    @staticmethod
    def _collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        max_len = max(ids.size(0) for ids in input_ids)

        input_ids_padded = torch.stack(
            [
                torch.cat(
                    [
                        ids,
                        torch.full(
                            (max_len - ids.size(0),),
                            50256,
                            dtype=ids.dtype,
                            device=ids.device,
                        ),
                    ]
                )
                for ids in input_ids
            ]
        )
        attention_mask_padded = torch.stack(
            [
                torch.cat(
                    [
                        mask,
                        torch.zeros(
                            max_len - mask.size(0), dtype=mask.dtype, device=mask.device
                        ),
                    ]
                )
                for mask in attention_mask
            ]
        )
        labels_padded = torch.stack(
            [
                torch.cat(
                    [
                        label,
                        torch.full(
                            (max_len - label.size(0),),
                            -100,
                            dtype=label.dtype,
                            device=label.device,
                        ),
                    ]
                )
                for label in labels
            ]
        )

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
        }


class TinyShakespeareDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: Tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset("tiny_shakespeare", split="train")
        self.text = self.dataset["text"]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        tokenized = self.tokenizer.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": tokenized["input_ids"].squeeze(0).clone(),
        }


def main():
    logger.info("Starting main function...")

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
        "use_mamba": True,
        "d_state": 16,
        "d_conv": 4,
        "expand_factor": 2.0,
        "dt_rank": 16,
        "dt_min": 0.001,
        "dt_max": 0.1,
        "sliding_window_size": 512,
    }

    logger.info("Initializing model...")
    model = UncertainTransformerLightningModule(hparams)

    logger.info("Loading tokenizer...")
    tokenizer = Tokenizer.from_pretrained("gpt2")
    model.tokenizer = tokenizer

    logger.info("Initializing DataModule...")
    datamodule = TinyShakespeareDataModule(
        tokenizer=tokenizer,
        max_length=hparams["max_length"],
        batch_size=hparams["batch_size"],
    )

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

    logger.info("Setting up TensorBoard logger...")
    tb_logger = TensorBoardLogger("logs", name="uncertain-transformer")

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

    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)

    logger.info("Training completed.")


if __name__ == "__main__":
    main()
