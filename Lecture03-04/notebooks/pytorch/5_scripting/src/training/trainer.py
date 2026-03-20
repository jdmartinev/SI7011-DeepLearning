"""Per-epoch training and evaluation loops."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    acc_metric: torchmetrics.Metric,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    acc_metric.reset()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        acc_metric.update(logits, labels)

    return total_loss / len(loader.dataset), acc_metric.compute().item()


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    acc_metric: torchmetrics.Metric,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate on a DataLoader. Returns (avg_loss, accuracy)."""
    model.eval()
    acc_metric.reset()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        acc_metric.update(logits, labels)

    return total_loss / len(loader.dataset), acc_metric.compute().item()
