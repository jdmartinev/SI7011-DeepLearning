# train.py
# ───────────────────────────────────────────────────────────────
# Entry point for training.
# Run with:   python train.py
# Or with a different config:  python train.py --config configs/config.yaml
#
# Responsibilities:
#   - Load config
#   - Set seed
#   - Build DataModule, Model, Callbacks, Logger, Trainer
#   - Call trainer.fit() and trainer.test()
# ───────────────────────────────────────────────────────────────

import argparse

import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger

from data  import MNISTDataModule
from model import MNISTClassifier


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_callbacks(cfg: dict):
    cb_cfg = cfg["callbacks"]

    early_stopping = EarlyStopping(
        monitor=cb_cfg["early_stopping"]["monitor"],
        patience=cb_cfg["early_stopping"]["patience"],
        mode=cb_cfg["early_stopping"]["mode"],
        verbose=True,
    )

    checkpoint = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="mnist-{epoch:02d}-{val_acc:.3f}",
        monitor=cb_cfg["checkpoint"]["monitor"],
        mode=cb_cfg["checkpoint"]["mode"],
        save_top_k=cb_cfg["checkpoint"]["save_top_k"],
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(
        logging_interval=cb_cfg["lr_monitor"]["logging_interval"]
    )

    return [early_stopping, checkpoint, lr_monitor], checkpoint


def main(config_path: str):
    cfg = load_config(config_path)

    # ── Reproducibility ────────────────────────────────────────
    pl.seed_everything(cfg["seed"], workers=True)

    # ── Data ───────────────────────────────────────────────────
    dm = MNISTDataModule(**cfg["data"])

    # ── Model ──────────────────────────────────────────────────
    model = MNISTClassifier(**cfg["model"])
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Callbacks ──────────────────────────────────────────────
    callbacks, checkpoint_cb = build_callbacks(cfg)

    # ── Logger ─────────────────────────────────────────────────
    logger = CSVLogger(**cfg["logger"])
    print(f"\nLogs → {logger.log_dir}")

    # ── Trainer ────────────────────────────────────────────────
    trainer = pl.Trainer(
        **cfg["trainer"],
        callbacks=callbacks,
        logger=logger,
    )

    # ── Fit ────────────────────────────────────────────────────
    trainer.fit(model, datamodule=dm)

    print(f"\nBest checkpoint : {checkpoint_cb.best_model_path}")
    print(f"Best val_acc    : {checkpoint_cb.best_model_score:.4f}")

    # ── Test ───────────────────────────────────────────────────
    # Always evaluate on the best checkpoint, not the last epoch
    results = trainer.test(model, datamodule=dm, ckpt_path="best")
    print("\nTest results:", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(args.config)
