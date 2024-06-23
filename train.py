import sys

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
from loguru import logger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    Callback
)
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.text import Perplexity
from transformers import get_cosine_schedule_with_warmup, GPT2Tokenizer

from core.data.datamodule import SlimPajamaDataModule
from core.models.uncertain_nn import (
    UncertainTransformerLMHeadModel,
    UncertainTransformerConfig,
)

# Configure Loguru
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

torch.set_float32_matmul_precision("high")


class LRRangeTestCallback(Callback):
    def __init__(self, min_lr=1e-7, max_lr=1.0, num_steps=100):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps

    def on_train_start(self, trainer, pl_module):
        """Runs the learning rate range test before training begins."""
        logger.info("Starting Learning Rate Range Test")
        pl_module.lr_range_test(
            trainer.datamodule, self.min_lr, self.max_lr, self.num_steps
        )
        logger.info("Learning Rate Range Test Complete")


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
        if hasattr(self.model, "gradient_checkpointing"):
            self.model.gradient_checkpointing = True
        elif hasattr(self.model, "transformer") and hasattr(
                self.model.transformer, "enable_gradient_checkpointing"
        ):
            self.model.transformer.enable_gradient_checkpointing()
        else:
            logger.warning("Gradient checkpointing not available for this model.")

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        self.perplexity = Perplexity(ignore_index=-100)

        self.tokenizer = None  # This should be set after initialization

        # Initialize state for streaming inference
        self.past_key_values = None

        # Learning rate range test
        self.lr_test_mode = False
        self.losses = []
        self.lrs = []

    def forward(
            self,
            input_ids,
            attention_mask=None,
            labels=None,
            past_key_values=None,
            use_cache=False,
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

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

        # Learning rate range test logging
        if self.lr_test_mode:
            self.losses.append(loss.item())
            lr_tensor = self.lr_schedulers().get_last_lr()[0]
            self.lrs.append(lr_tensor)

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
            generated_text = self.tokenizer.decode(
                generated[0], skip_special_tokens=True
            )
            self.logger.experiment.add_text(
                "generated_text", generated_text, self.current_epoch
            )

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
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def stream_generate(self, input_ids: torch.Tensor, max_length: int = 50, **kwargs):
        """
        Generates text in a streaming fashion, processing one token at a time.

        Args:
            input_ids (torch.Tensor): The initial input token IDs.
            max_length (int, optional): The maximum length of the generated sequence. Defaults to 50.
            **kwargs: Additional keyword arguments for the `model.generate` method.

        Returns:
            torch.Tensor: The generated token IDs.
        """
        for _ in range(max_length):
            outputs = self.forward(
                input_ids[:, -1:],
                past_key_values=self.past_key_values,
                use_cache=True,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            self.past_key_values = outputs.past_key_values
        return input_ids

    def lr_range_test(self, datamodule, min_lr=1e-7, max_lr=1.0, num_steps=100):
        """Performs a learning rate range test."""

        self.lr_test_mode = True

        # Reset losses and learning rates
        self.losses = []
        self.lrs = []

        # Use a simple one-cycle LR scheduler
        optimizer = self.configure_optimizers()["optimizer"]
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=num_steps,
            pct_start=0.1,  # Warmup for 10% of steps
            anneal_strategy="linear",
        )
        self.lr_schedulers = lr_scheduler  # Needed by PyTorch Lightning

        # Manual training loop for LR range test
        iterator = iter(datamodule.train_dataloader())
        for i in range(num_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(datamodule.train_dataloader())
                batch = next(iterator)

            self.training_step(batch, i)
            lr_scheduler.step()  # Update learning rate

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(self.lrs, self.losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Loss")
        plt.title("Learning Rate Range Test")
        plt.show()

        self.lr_test_mode = False  # Turn off test mode


def main():
    logger.info("Starting main function...")
    hparams = {
        "vocab_size": 50257,
        "d_model": 512,
        "n_heads": 8,
        "d_ff": 2048,
        "n_layers": 6,
        "dropout": 0.1,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_epochs": 10,
        "pad_token_id": 50256,
        "max_length": 1024,
        "subset_size": 0.1,
        "accumulate_grad_batches": 4,
        "gradient_clip_val": 1.0,
        "val_check_interval": 1.0,
        "use_stable_embedding": True,
        "num_groups": 16,
        "cema_hidden_dim": 32,
        "z_dim": 512,
        "v_dim": 512,
        "chunk_size": 64,
        "use_gelu_approximation": True,
        # New Mamba-specific parameters
        "d_state": 16,
        "d_conv": 4,
        "mamba_expand_factor": 2,
        "mamba_dt_rank": None,
        "mamba_dt_min": 0.001,
        "mamba_dt_max": 0.1,
    }

    logger.info("Initializing model...")
    model = UncertainTransformerLightningModule(hparams)

    logger.info("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model.tokenizer = tokenizer

    logger.info("Initializing DataModule...")
    try:
        datamodule = SlimPajamaDataModule(
            tokenizer=tokenizer,
            max_length=hparams["max_length"],
            batch_size=hparams["batch_size"],
            streaming=True,  # Enable streaming in the DataModule
        )
        logger.success("DataModule initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing DataModule: {str(e)}")
        raise

    logger.info("Setting up callbacks...")
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

    logger.info("Setting up TensorBoard logger...")
    tb_logger = TensorBoardLogger("logs", name="uncertain-transformer")

    logger.info("Setting up trainer...")
    trainer = pl.Trainer(
        max_epochs=hparams["max_epochs"],
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=tb_logger,
        accumulate_grad_batches=hparams["accumulate_grad_batches"],
        precision="16-mixed",
        gradient_clip_val=hparams["gradient_clip_val"],
        val_check_interval=hparams["val_check_interval"],
        deterministic=True,
        log_every_n_steps=10,
    )


    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)
    logger.success("Training completed.")

    # Example streaming generation
    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    generated_ids = model.stream_generate(input_ids, max_length=50)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    logger.info(f"Generated text (streaming): {generated_text}")


if __name__ == "__main__":
    main()
