"""Checkpoint save / load utilities."""
from __future__ import annotations

import os
import glob
import torch
import torch.nn as nn
from typing import Any


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    save_dir: str,
    run_name: str,
    save_top_k: int = 2,
) -> str:
    """Save a checkpoint and prune old ones, keeping only the top-k by val_acc."""
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{run_name}_epoch{epoch:03d}_acc{val_acc:.4f}.pt"
    path = os.path.join(save_dir, filename)

    torch.save(
        {
            "epoch": epoch,
            "val_acc": val_acc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )

    # Prune: keep only the save_top_k checkpoints with highest val_acc
    pattern = os.path.join(save_dir, f"{run_name}_epoch*.pt")
    checkpoints = sorted(
        glob.glob(pattern),
        key=lambda p: float(p.rsplit("_acc", 1)[-1].replace(".pt", "")),
        reverse=True,
    )
    for old in checkpoints[save_top_k:]:
        os.remove(old)

    return path


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    """Load a checkpoint into model (and optionally optimizer)."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def best_checkpoint(save_dir: str, run_name: str) -> str | None:
    """Return the path to the checkpoint with the highest val_acc."""
    pattern = os.path.join(save_dir, f"{run_name}_epoch*.pt")
    checkpoints = sorted(
        glob.glob(pattern),
        key=lambda p: float(p.rsplit("_acc", 1)[-1].replace(".pt", "")),
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None
