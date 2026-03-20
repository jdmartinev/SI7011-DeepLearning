"""Full test-set evaluation: accuracy, per-class report, confusion matrix."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchmetrics
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.data.dataset import CLASSES


@torch.inference_mode()
def evaluate_test_set(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Run the model over the test loader and return a results dict.

    Returns
    -------
    dict with keys:
        - accuracy (float)
        - report   (str) — sklearn classification_report
        - confusion_matrix (np.ndarray, shape [10, 10])
        - all_preds  (list[int])
        - all_labels (list[int])
    """
    model.eval()
    acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(CLASSES)).to(device)

    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds  = logits.argmax(dim=1)

        acc_metric.update(logits, labels)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    accuracy = acc_metric.compute().item()
    report   = classification_report(all_labels, all_preds, target_names=CLASSES, digits=4)
    cm       = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy":         accuracy,
        "report":           report,
        "confusion_matrix": cm,
        "all_preds":        all_preds,
        "all_labels":       all_labels,
    }
