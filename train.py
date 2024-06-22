import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner import Tuner
from torchmetrics.text import Perplexity
from transformers import get_cosine_schedule_with_warmup, GPT2Tokenizer

from core.data.datamodule import SlimPajamaDataModule
from core.models.uncertain_nn import UncertainTransformerLMHeadModel, UncertainTransformerConfig

torch.set_float32_matmul_precision('high')


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
            use_stable_embedding=hparams["use_stable_embedding"],
            num_groups=hparams["num_groups"],
            cema_hidden_dim=hparams["cema_hidden_dim"],
            z_dim=hparams["z_dim"],
            v_dim=hparams["v_dim"],
            chunk_size=hparams["chunk_size"],
        )
        self.model = UncertainTransformerLMHeadModel(config)

        # Enable gradient checkpointing if the model supports it
        if hasattr(self.model, 'gradient_checkpointing'):
            self.model.gradient_checkpointing = True
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'enable_gradient_checkpointing'):
            self.model.transformer.enable_gradient_checkpointing()
        else:
            print("Warning: Gradient checkpointing not available for this model.")

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        self.perplexity = Perplexity(ignore_index=-100)

        self.tokenizer = None  # This should be set after initialization

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        attention_mask = (input_ids != self.hparams["pad_token_id"]).long()
        outputs = self.forward(input_ids, attention_mask=attention_mask, labels=labels)
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

        # Use the Lightning accumulate_grad_batches instead of manual implementation
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
                    p
                    for n, p in param_optimizer
                    if all(nd not in n for nd in no_decay)
                ],
                "weight_decay": self.hparams["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
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


def main():
    hparams = {
        "vocab_size": 50257,
        "d_model": 512,
        "n_heads": 8,
        "d_ff": 2048,
        "n_layers": 6,
        "dropout": 0.1,
        "batch_size": 16,
        "learning_rate": 3e-4,  # Initial learning rate, will be updated by LR finder
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_epochs": 10,
        "pad_token_id": 50256,
        "max_length": 1024,
        "subset_size": 0.1,
        "accumulate_grad_batches": 4,
        "gradient_clip_val": 1.0,
        "val_check_interval": 0.25,
        "use_stable_embedding": True,
        "num_groups": 16,
        "cema_hidden_dim": 32,
        "z_dim": 512,
        "v_dim": 512,
        "chunk_size": 64,
        "use_gelu_approximation": True,
    }

    model = UncertainTransformerLightningModule(hparams)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    datamodule = SlimPajamaDataModule(
        tokenizer=tokenizer,
        max_length=hparams["max_length"],
        train_num_examples=100000,
        val_num_examples=10000,
        test_num_examples=10000,
        batch_size=hparams["batch_size"],
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="uncertain-transformer-{epoch:02d}-{val_loss:.2f}-{val_perplexity:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        every_n_epochs=1,
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    lr_monitor = LearningRateMonitor(logging_interval="step")

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

    # Learning rate finder
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, datamodule=datamodule)
    if lr_finder.suggestion() is not None:
        new_lr = lr_finder.suggestion()
        print(f"Suggested Learning Rate: {new_lr}")
        model.hparams.learning_rate = new_lr
    else:
        print("Learning rate finder failed to suggest a learning rate. Using default.")

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
