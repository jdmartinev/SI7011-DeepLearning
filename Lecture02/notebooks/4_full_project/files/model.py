# model.py
# ───────────────────────────────────────────────────────────────
# LightningModule for MNIST classification.
# Responsible for: architecture, loss, optimizer, scheduler,
# and metric logging. Nothing about data or training loops.
# ───────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class MNISTClassifier(pl.LightningModule):
    """
    Feed-forward classifier for MNIST.

    Architecture:
        Input(784) → FC(hidden) → ReLU → Dropout
                   → FC(hidden/2) → ReLU → Dropout
                   → FC(num_classes)

    Args:
        input_size  : flattened image size (28*28 = 784)
        hidden_size : units in the first hidden layer
        num_classes : number of output classes (10 for MNIST)
        dropout     : dropout probability
        lr          : initial learning rate
    """

    def __init__(
        self,
        input_size=784,
        hidden_size=256,
        num_classes=10,
        dropout=0.3,
        lr=1e-3,
    ):
        super().__init__()
        # Stores all args in self.hparams and saves them to checkpoints.
        # This allows load_from_checkpoint() to reconstruct the model
        # without needing to pass arguments manually.
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),  # raw logits — no softmax here
        )

    # ── Forward ────────────────────────────────────────────────
    def forward(self, x):
        """Used for inference: model(x). Returns raw logits."""
        return self.model(x)

    # ── Training ───────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        # on_step=True  → logs a value for every batch (useful to spot instability)
        # on_epoch=True → also logs the epoch-level average
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc",  acc,  on_step=False, on_epoch=True, prog_bar=True)
        return loss  # Lightning calls .backward() on this

    # ── Validation ─────────────────────────────────────────────
    def validation_step(self, batch, batch_idx):
        # Lightning automatically sets eval() mode and disables gradients
        loss, acc = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc",  acc,  prog_bar=True)

    # ── Test ───────────────────────────────────────────────────
    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("test_loss", loss)
        self.log("test_acc",  acc)

    # ── Optimizer + Scheduler ──────────────────────────────────
    def configure_optimizers(self):
        """
        Return optimizer and scheduler configuration.
        Lightning calls this once before training starts.

        ReduceLROnPlateau: halves the LR when val_loss hasn't improved
        for `patience` epochs. Great default for most tasks.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # metric the scheduler watches
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ── Shared logic ───────────────────────────────────────────
    def _shared_step(self, batch):
        """Compute loss and accuracy — shared across train/val/test."""
        x, y   = batch
        logits = self(x)
        loss   = F.cross_entropy(logits, y)
        preds  = torch.argmax(logits, dim=1)
        acc    = (preds == y).float().mean()
        return loss, acc
