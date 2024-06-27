# train.py

import os
import sys
from typing import Any, Optional
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets import load_dataset
from loguru import logger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset

from core.models.uncertainty.uncertainty import (
    UncertainTransformerLMHeadModel,
    UncertainTransformerConfig, )
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
torch.autograd.set_detect_anomaly(True)


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

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Optional[torch.Tensor]:
        outputs = self(**batch)
        loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            self.log("nan_loss", 1.0, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            return None

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        if outputs is None:
            self.zero_grad()
            self.trainer.strategy.optimizer_step(self.optimizers())

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

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
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

        return {
            "val_loss": loss,
            "val_perplexity": perplexity,
        }


class TinyShakespeareDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: Tokenizer, max_length: int = 1024, split: str = "train"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset("tiny_shakespeare", split=split, trust_remote_code=True)
        self.text = self.dataset["text"][0]  # Access the text from the single example
        self.lines = self.text.split('\n')
        self.tokenized_lines = self._tokenize_lines()

    def _tokenize_lines(self):
        tokenized = []
        for line in self.lines:
            if line.strip():  # Skip empty lines
                tokens = self.tokenizer.encode(line, add_special_tokens=False)
                tokenized.append(tokens)
        return tokenized

    def __len__(self):
        return len(self.tokenized_lines)

    def __getitem__(self, idx):
        tokens = self.tokenized_lines[idx]
        tokens = tokens[:self.max_length]  # Truncate to max_length

        input_ids = torch.tensor(tokens[:-1])
        labels = torch.tensor(tokens[1:])

        # Pad sequences
        input_ids = F.pad(input_ids, (0, self.max_length - len(input_ids)), value=self.tokenizer.pad_token_id)
        labels = F.pad(labels, (0, self.max_length - len(labels)), value=-100)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class TinyShakespeareDataModule(pl.LightningDataModule):
    def __init__(
            self,
            tokenizer: Tokenizer,
            max_length: int = 1024,
            batch_size: int = 32,
            num_workers: int = 4,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset: Optional[TinyShakespeareDataset] = None
        self.val_dataset: Optional[TinyShakespeareDataset] = None
        self.test_dataset: Optional[TinyShakespeareDataset] = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = TinyShakespeareDataset(
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                split="train",
            )
            self.val_dataset = TinyShakespeareDataset(
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                split="validation"
            )

        if stage == "test" or stage is None:
            self.test_dataset = TinyShakespeareDataset(
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                split="test"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True
        )


class Wikitext2Dataset(Dataset):
    def __init__(self, tokenizer, data, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self.preprocess_data(data)

    def preprocess_data(self, data):
        examples = []
        for item in data:
            if item['text'].strip():  # Skip empty texts
                encodings = self.tokenizer(item['text'], truncation=True, max_length=self.max_length,
                                           padding='max_length')
                examples.append({
                    'input_ids': torch.tensor(encodings['input_ids']),
                    'attention_mask': torch.tensor(encodings['attention_mask']),
                })
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        labels = item['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'labels': labels
        }


class Wikitext2DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=32, max_length=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def setup(self, stage=None):
        dataset = load_dataset("wikitext", "wikitext-2-v1")

        self.train_dataset = Wikitext2Dataset(self.tokenizer, dataset['train'], self.max_length)
        self.val_dataset = Wikitext2Dataset(self.tokenizer, dataset['validation'], self.max_length)
        self.test_dataset = Wikitext2Dataset(self.tokenizer, dataset['test'], self.max_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True)


def main():
    logger.info("Starting main function...")

    hparams = {
        "vocab_size": 50257,
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 512,
        "n_layers": 2,
        "dropout": 0.1,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "max_length": 128,
        "batch_size": 32,
        "accumulate_grad_batches": 1,
        "max_epochs": 10,
        "pad_token_id": 50256,
        "use_mamba": True,  # Set to True if you want to use Mamba
        "d_state": 8,
        "d_conv": 2,
        "expand_factor": 1.5,
        "dt_rank": 8,
        "dt_min": 0.001,
        "dt_max": 0.1,
        "sliding_window_size": 128,
        "use_gradient_checkpointing": False,
    }

    logger.info("Initializing model...")
    model = UncertainTransformerLightningModule(hparams)

    logger.info("Loading tokenizer...")
    tokenizer = Tokenizer.from_pretrained("gpt2")
    model.tokenizer = tokenizer

    logger.info("Initializing DataModule...")
    datamodule = Wikitext2DataModule(
        tokenizer=tokenizer,
        batch_size=hparams["batch_size"],
        max_length=hparams["max_length"],
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
        log_every_n_steps=1,
        val_check_interval=0.25,
        detect_anomaly=True,
        num_sanity_val_steps=0
    )

    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)

    logger.info("Training completed.")


if __name__ == "__main__":
    main()
