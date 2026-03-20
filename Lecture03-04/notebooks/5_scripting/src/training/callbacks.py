"""Training callbacks: checkpoint saving and early stopping."""
from __future__ import annotations

import torch
import torch.nn as nn
from src.utils.io import save_checkpoint


class ModelCheckpoint:
    """Save the model whenever val_acc improves, keeping the top-k checkpoints."""

    def __init__(self, save_dir: str, run_name: str, save_top_k: int = 2):
        self.save_dir  = save_dir
        self.run_name  = run_name
        self.save_top_k = save_top_k
        self.best_acc  = -1.0
        self.last_path: str | None = None

    def step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_acc: float,
    ) -> bool:
        """Returns True if a new checkpoint was saved."""
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.last_path = save_checkpoint(
                model, optimizer, epoch, val_acc,
                self.save_dir, self.run_name, self.save_top_k,
            )
            return True
        return False


class EarlyStopping:
    """Stop training when val_acc has not improved for ``patience`` epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_acc  = -1.0
        self.counter   = 0

    @property
    def should_stop(self) -> bool:
        return self.counter >= self.patience

    def step(self, val_acc: float) -> None:
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter  = 0
        else:
            self.counter += 1
