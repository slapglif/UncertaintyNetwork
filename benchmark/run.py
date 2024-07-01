# run.py

import os
import sys
import argparse
from typing import Dict, Any, List
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from rich.console import Console
from rich.panel import Panel
from loguru import logger
from  benchmark.engine.data_utils import Wikitext2DataModule, Wikitext2Dataset
from core.models.uncertainty.uncertainty import UncertainTransformerConfig
from train import UncertainTransformerLightningModule

console = Console()


def setup_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate UncertainTransformer on WikiText-2")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for resuming training")
    return parser


def load_config(config_path: str) -> Dict[str, Any]:
    import json
    with open(config_path, 'r') as f:
        return json.load(f)


def setup_data_module(config: Dict[str, Any]) -> pl.LightningDataModule:
    class WikiText2DataModule(pl.LightningDataModule):
        def __init__(self, config: Dict[str, Any]):
            super().__init__()
            self.config = config
            self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])

        def setup(self, stage=None):
            dataset = load_dataset("wikitext", "wikitext-2-v1")
            self.train_dataset = Wikitext2Dataset(dataset['train'], self.tokenizer, self.config['max_length'])
            self.val_dataset = Wikitext2Dataset(dataset['validation'], self.tokenizer, self.config['max_length'])
            self.test_dataset = Wikitext2Dataset(dataset['test'], self.tokenizer, self.config['max_length'])

        def train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=self.config['batch_size'],
                              num_workers=self.config['num_workers'], shuffle=True)

        def val_dataloader(self):
            return DataLoader(self.val_dataset, batch_size=self.config['batch_size'],
                              num_workers=self.config['num_workers'])

        def test_dataloader(self):
            return DataLoader(self.test_dataset, batch_size=self.config['batch_size'],
                              num_workers=self.config['num_workers'])

    return WikiText2DataModule(config)


def setup_callbacks(config: Dict[str, Any]) -> List[pl.Callback]:
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['checkpoint_dir'],
        filename='uncertain_transformer-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config['early_stopping_patience'],
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    return [checkpoint_callback, early_stop_callback, lr_monitor]


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    config = load_config(args.config)

    console.print(Panel.fit("🚀 [bold cyan]Training UncertainTransformer on WikiText-2[/bold cyan] 🚀"))

    # Setup logging
    logger.remove()
    logger.add(sys.stderr,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.add(f"{config['log_dir']}/train.log", rotation="1 day")

    # Setup data module
    data_module = setup_data_module(config)

    # Setup model
    if args.checkpoint:
        console.print(f"Loading model from checkpoint: {args.checkpoint}")
        model = UncertainTransformerLightningModule.load_from_checkpoint(args.checkpoint)
    else:
        model = UncertainTransformerLightningModule(config)

    # Setup trainer
    callbacks = setup_callbacks(config)
    tb_logger = TensorBoardLogger(save_dir=config['log_dir'], name='uncertain_transformer')

    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        callbacks=callbacks,
        logger=tb_logger,
        precision="16-mixed" if config['use_mixed_precision'] else 32,
        accumulate_grad_batches=config['accumulate_grad_batches'],
        gradient_clip_val=config['gradient_clip_val'],
        val_check_interval=config['val_check_interval'],
        log_every_n_steps=config['log_every_n_steps'],
    )

    # Train model
    console.print("Starting training...")
    trainer.fit(model, data_module)

    # Test model
    console.print("Starting testing...")
    test_result = trainer.test(model, data_module)

    console.print(Panel.fit("[bold green]Training and evaluation completed![/bold green]"))
    console.print("Test results:")
    console.print(test_result)

    # Save final model
    final_model_path = os.path.join(config['model_dir'], 'final_model.ckpt')
    trainer.save_checkpoint(final_model_path)
    console.print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()