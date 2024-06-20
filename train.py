import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from rouge_score import rouge_scorer
from torchmetrics.text import Perplexity

from core.data.datamodule import SlimPajamaDataModule
from core.models.uncertain_nn import UncertainTransformer

torch.set_float32_matmul_precision('medium')


class UncertainTransformerLightningModule(pl.LightningModule):
    def __init__(self, hparams: dict, datamodule: SlimPajamaDataModule):
        """
        Initializes the UncertainTransformerLightningModule.

        Args:
            hparams (dict): A dictionary containing the hyperparameters.
            datamodule (SlimPajamaDataModule): The data module instance.
        """
        super().__init__()
        self.save_hyperparameters(hparams)  # Use save_hyperparameters to store hparams
        self.data_module = datamodule
        self.model = UncertainTransformer(
            vocab_size=self.hparams["vocab_size"],
            d_model=self.hparams["d_model"],
            n_heads=self.hparams["n_heads"],
            d_ff=self.hparams["d_ff"],
            n_layers=self.hparams["n_layers"],
            dropout=self.hparams["dropout"],
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.hparams["pad_token_id"])
        self.perplexity = Perplexity(ignore_index=self.hparams["pad_token_id"])
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def forward(self, src, src_mask):
        return self.model(src, src_mask)

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        src_mask = self.create_src_mask(src)
        output = self.model(src, src_mask)
        loss = self.criterion(output.view(-1, output.size(-1)), tgt.contiguous().view(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        src_mask = self.create_src_mask(src)
        output = self.model(src, src_mask)
        loss = self.criterion(output.view(-1, output.size(-1)), tgt.contiguous().view(-1))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        # Generate text using the model's output
        generated_ids = self.model.generate(src, max_length=tgt.size(1), num_return_sequences=1)
        generated_text = [self.model.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]

        # Calculate perplexity
        perplexity = self.perplexity(output.view(-1, output.size(-1)), tgt.contiguous().view(-1))
        self.log("val_perplexity", perplexity, on_epoch=True, prog_bar=True, logger=True)

        # Calculate ROUGE-L
        tgt_text = [self.model.tokenizer.decode(ids, skip_special_tokens=True) for ids in tgt]
        rouge_scores = self.rouge_scorer.score(" ".join(generated_text), " ".join(tgt_text))
        self.log("val_rouge_l", rouge_scores["rougeL"].fmeasure, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["learning_rate"])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams["learning_rate"],
            steps_per_epoch=len(self.train_dataloader()),
            epochs=self.hparams["max_epochs"],
            anneal_strategy='linear',
            pct_start=0.1,
            div_factor=25.0,
            final_div_factor=1e4,
        )
        return [optimizer], [scheduler]

    def create_src_mask(self, src):
        src_mask = torch.tensor(src != self.hparams["pad_token_id"], dtype=torch.float).unsqueeze(1).unsqueeze(2)
        return src_mask

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
        "precision": 16,
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
