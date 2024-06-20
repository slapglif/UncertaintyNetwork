# core/scripts/train.py
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from rouge_score import rouge_scorer
from torch import nn
from torchmetrics.text import Perplexity

from core.data.datamodule import SlimPajamaDataModule
from core.models.uncertain_nn import UncertainTransformerLMHeadModel, UncertainTransformerConfig

torch.set_float32_matmul_precision('medium')


class UncertainTransformerLightningModule(pl.LightningModule):
    def __init__(self, hparams: dict, datamodule: SlimPajamaDataModule):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = datamodule

        config = UncertainTransformerConfig(
            vocab_size=hparams["vocab_size"],
            d_model=hparams["d_model"],
            n_heads=hparams["n_heads"],
            d_ff=hparams["d_ff"],
            n_layers=hparams["n_layers"],
            dropout=hparams["dropout"],
            max_position_embeddings=hparams["max_length"],
        )
        self.model = UncertainTransformerLMHeadModel(config)

        self.criterion = nn.CrossEntropyLoss(ignore_index=hparams["pad_token_id"])
        self.perplexity = Perplexity(ignore_index=hparams["pad_token_id"])
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        attention_mask = (input_ids != self.hparams["pad_token_id"]).long()
        outputs = self.forward(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        attention_mask = (input_ids != self.hparams["pad_token_id"]).long()
        outputs = self.forward(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()


def main():
    hparams = {
        "vocab_size": 50257,
        "d_model": 512,
        "n_heads": 8,
        "d_ff": 2048,
        "n_layers": 6,
        "dropout": 0.1,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "max_epochs": 10,
        "pad_token_id": 0,
        "max_length": 1024,
        "subset_size": 0.1,
        "accumulate_grad_batches": 1,
        "precision": "16-mixed",
        "auto_lr_find": True,
        "auto_scale_batch_size": "binsearch",
        "gradient_clip_val": 1.0,
        "val_check_interval": 0.25,
    }

    datamodule = SlimPajamaDataModule(
        batch_size=hparams["batch_size"],
        subset_size=hparams["subset_size"],
        max_length=hparams["max_length"],
    )

    model = UncertainTransformerLightningModule(hparams, datamodule)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="uncertain-transformer-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger = TensorBoardLogger("logs", name="uncertain-transformer")

    trainer = pl.Trainer(
        max_epochs=hparams["max_epochs"],
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        accumulate_grad_batches=hparams["accumulate_grad_batches"],
        precision=hparams["precision"],
        gradient_clip_val=hparams["gradient_clip_val"],
        val_check_interval=hparams["val_check_interval"],
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
