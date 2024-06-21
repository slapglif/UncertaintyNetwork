import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torchmetrics.text import Perplexity
from transformers import get_linear_schedule_with_warmup

from core.data.datamodule import SlimPajamaDataModule
from core.models.uncertain_nn import UncertainTransformerLMHeadModel, UncertainTransformerConfig

torch.set_float32_matmul_precision('medium')


class UncertainTransformerLightningModule(pl.LightningModule):
    def __init__(self, hparams: dict):
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
            use_rezero=hparams["use_rezero"],
            use_gelu_approximation=hparams["use_gelu_approximation"],
            use_stable_embedding=hparams["use_stable_embedding"],
        )
        self.model = UncertainTransformerLMHeadModel(config)
        self.model.enable_gradient_checkpointing()

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        self.perplexity = Perplexity(ignore_index=-100)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        attention_mask = (input_ids != self.hparams["pad_token_id"]).long()
        outputs = self.forward(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Invalid loss detected at batch {batch_idx}")
            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Attention mask shape: {attention_mask.shape}")
            print(f"Logits shape: {outputs.logits.shape}")
            print(f"Logits min: {outputs.logits.min()}, max: {outputs.logits.max()}")
            return None

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        attention_mask = (input_ids != self.hparams["pad_token_id"]).long()
        outputs = self.forward(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        perplexity = self.perplexity(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
        self.log("val_perplexity", perplexity, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams["weight_decay"],
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
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
        scheduler = get_linear_schedule_with_warmup(
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


def main():
    hparams = {
        "vocab_size": 50257,
        "d_model": 768,
        "n_heads": 12,
        "d_ff": 3072,
        "n_layers": 12,
        "dropout": 0.1,
        "batch_size": 32,
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_epochs": 10,
        "pad_token_id": 50256,
        "max_length": 1024,
        "subset_size": 0.1,
        "accumulate_grad_batches": 4,
        "gradient_clip_val": 1.0,
        "val_check_interval": 0.25,
        "use_rezero": True,
        "use_gelu_approximation": True,
        "use_stable_embedding": True,
    }

    model = UncertainTransformerLightningModule(hparams)

    datamodule = SlimPajamaDataModule(
        batch_size=hparams["batch_size"],
        subset_size=hparams["subset_size"],
        max_length=hparams["max_length"],
    )

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
        precision="16-mixed",
        gradient_clip_val=hparams["gradient_clip_val"],
        val_check_interval=hparams["val_check_interval"],
        deterministic=True,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()