#!/usr/bin/env python
"""Evaluation pipeline: run a trained checkpoint against the CIFAR-10 test set.

Usage
-----
    python scripts/evaluate.py --checkpoint runs/resnet18-two-stage_epoch050_acc0.9312.pt
    python scripts/evaluate.py --checkpoint runs/best.pt --config config/base.yaml
    python scripts/evaluate.py --checkpoint runs/best.pt --save-report results/
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a ResNet-18 checkpoint on CIFAR-10 test set")
    p.add_argument("--checkpoint",   required=True, help="Path to .pt checkpoint file")
    p.add_argument("--config",       default="config/base.yaml")
    p.add_argument("--data-dir",     default=None,  help="Override data.root from config")
    p.add_argument("--batch-size",   type=int, default=None)
    p.add_argument("--save-report",  default=None,  help="Directory to save JSON report and confusion matrix")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.data_dir:   cfg["data"]["root"] = args.data_dir
    if args.batch_size: cfg["training"]["batch_size"] = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")

    # ── Model ──────────────────────────────────────────────────────────
    from src.models.resnet import build_model
    from src.utils.io import load_checkpoint

    model = build_model(num_classes=cfg["data"]["num_classes"])
    ckpt  = load_checkpoint(args.checkpoint, model)
    model.to(device)

    trained_epoch = ckpt.get("epoch", "?")
    saved_val_acc = ckpt.get("val_acc", "?")
    print(f"Loaded     : epoch {trained_epoch}  |  saved val acc {saved_val_acc}")

    # ── Data ──────────────────────────────────────────────────────────
    from src.data.dataset import build_dataloaders

    _, _, test_loader = build_dataloaders(
        root=cfg["data"]["root"],
        val_split=cfg["data"]["val_split"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        seed=cfg["data"]["seed"],
    )

    # ── Evaluate ───────────────────────────────────────────────────────
    from src.evaluation.metrics import evaluate_test_set

    results = evaluate_test_set(model, test_loader, device)

    print(f"\nTest Accuracy : {results['accuracy']:.4f}  ({results['accuracy']*100:.2f}%)")
    print("\nClassification Report:")
    print(results["report"])

    # ── Optional: save report ─────────────────────────────────────────
    if args.save_report:
        out_dir = Path(args.save_report)
        out_dir.mkdir(parents=True, exist_ok=True)

        report_path = out_dir / "report.json"
        report_data = {
            "checkpoint":  args.checkpoint,
            "epoch":       trained_epoch,
            "val_acc":     saved_val_acc,
            "test_accuracy": results["accuracy"],
            "all_preds":   results["all_preds"],
            "all_labels":  results["all_labels"],
        }
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)

        cm_path = out_dir / "confusion_matrix.npy"
        np.save(cm_path, results["confusion_matrix"])

        print(f"\nReport saved to      : {report_path}")
        print(f"Confusion matrix to  : {cm_path}")


if __name__ == "__main__":
    main()
