#!/usr/bin/env python
"""Training pipeline: two-stage transfer learning on CIFAR-10.

Usage
-----
    python scripts/train.py
    python scripts/train.py --config config/base.yaml
    python scripts/train.py --config config/base.yaml --stage1-epochs 3 --stage2-epochs 10
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn
import torchmetrics
import yaml

from src.data.dataset import build_dataloaders
from src.models.resnet import build_model, unfreeze_backbone, count_parameters
from src.training.trainer import train_one_epoch, evaluate
from src.training.callbacks import ModelCheckpoint, EarlyStopping
from src.utils.logging import get_writer

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage ResNet-18 fine-tuning on CIFAR-10")
    p.add_argument("--config",         default="config/base.yaml")
    p.add_argument("--stage1-epochs",  type=int,   default=None, help="Override config stage1.epochs")
    p.add_argument("--stage2-epochs",  type=int,   default=None, help="Override config stage2.epochs")
    p.add_argument("--batch-size",     type=int,   default=None, help="Override config batch_size")
    p.add_argument("--run-name",       type=str,   default=None, help="Override config run_name")
    p.add_argument("--data-dir",       type=str,   default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.stage1_epochs: cfg["training"]["stage1"]["epochs"] = args.stage1_epochs
    if args.stage2_epochs: cfg["training"]["stage2"]["epochs"] = args.stage2_epochs
    if args.batch_size:    cfg["training"]["batch_size"] = args.batch_size
    if args.run_name:      cfg["logging"]["run_name"] = args.run_name
    if args.data_dir:      cfg["data"]["root"] = args.data_dir

    # ── Setup ──────────────────────────────────────────────────────────
    torch.manual_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    from src.data.dataset import build_dataloaders
    train_loader, val_loader, _ = build_dataloaders(
        root=cfg["data"]["root"],
        val_split=cfg["data"]["val_split"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        seed=cfg["data"]["seed"],
    )

    # ── Model ──────────────────────────────────────────────────────────
    from src.models.resnet import build_model, unfreeze_backbone, count_parameters
    model = build_model(
        num_classes=cfg["data"]["num_classes"],
        weights=cfg["model"]["weights"],
    ).to(device)

    params = count_parameters(model)
    print(f"Parameters — total: {params['total']:,}  trainable: {params['trainable']:,}  frozen: {params['frozen']:,}")

    # ── Logging & callbacks ───────────────────────────────────────────
    from src.utils.logging import get_writer
    from src.training.callbacks import ModelCheckpoint, EarlyStopping

    writer  = get_writer(cfg["logging"]["log_dir"], cfg["logging"]["run_name"])
    ckpt_cb = ModelCheckpoint(
        cfg["checkpoint"]["save_dir"],
        cfg["logging"]["run_name"],
        cfg["checkpoint"]["save_top_k"],
    )
    early_cb = EarlyStopping(patience=10)

    # ── Shared criterion & metrics ────────────────────────────────────
    from src.training.trainer import train_one_epoch, evaluate

    criterion = nn.CrossEntropyLoss()
    train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=cfg["data"]["num_classes"]).to(device)
    val_acc   = torchmetrics.Accuracy(task="multiclass", num_classes=cfg["data"]["num_classes"]).to(device)

    # ── Stage 1: head only ────────────────────────────────────────────
    s1 = cfg["training"]["stage1"]
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=s1["lr"], weight_decay=s1["weight_decay"],
    )

    print(f"\n--- Stage 1: FC head only ({s1['epochs']} epochs, lr={s1['lr']}) ---")
    for epoch in range(1, s1["epochs"] + 1):
        tr_loss, tr_acc_val = train_one_epoch(model, train_loader, optimizer, criterion, train_acc, device)
        vl_loss, vl_acc_val = evaluate(model, val_loader, criterion, val_acc, device)

        writer.add_scalars("Loss",     {"train": tr_loss, "val": vl_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": tr_acc_val, "val": vl_acc_val}, epoch)
        writer.add_scalar("Stage", 1, epoch)

        saved = ckpt_cb.step(model, optimizer, epoch, vl_acc_val)
        early_cb.step(vl_acc_val)

        print(
            f"  Epoch {epoch:02d}/{s1['epochs']} | "
            f"train loss {tr_loss:.4f}  acc {tr_acc_val:.4f} | "
            f"val loss {vl_loss:.4f}  acc {vl_acc_val:.4f}"
            + (" ✓ saved" if saved else "")
        )

        if early_cb.should_stop:
            print("  Early stopping triggered.")
            break

    # ── Stage 2: full fine-tuning ─────────────────────────────────────
    unfreeze_backbone(model)
    params = count_parameters(model)
    s2 = cfg["training"]["stage2"]
    print(f"\n--- Stage 2: full fine-tuning ({s2['epochs']} epochs, lr={s2['lr']}) ---")
    print(f"  All {params['trainable']:,} parameters now trainable.")
    optimizer = torch.optim.Adam(model.parameters(), lr=s2["lr"], weight_decay=s2["weight_decay"])
    early_cb  = EarlyStopping(patience=10)       # reset patience for stage 2

    total_epochs = s1["epochs"] + s2["epochs"]
    for epoch in range(s1["epochs"] + 1, total_epochs + 1):
        tr_loss, tr_acc_val = train_one_epoch(model, train_loader, optimizer, criterion, train_acc, device)
        vl_loss, vl_acc_val = evaluate(model, val_loader, criterion, val_acc, device)

        writer.add_scalars("Loss",     {"train": tr_loss, "val": vl_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": tr_acc_val, "val": vl_acc_val}, epoch)
        writer.add_scalar("Stage", 2, epoch)

        saved = ckpt_cb.step(model, optimizer, epoch, vl_acc_val)
        early_cb.step(vl_acc_val)

        print(
            f"  Epoch {epoch:02d}/{total_epochs} | "
            f"train loss {tr_loss:.4f}  acc {tr_acc_val:.4f} | "
            f"val loss {vl_loss:.4f}  acc {vl_acc_val:.4f}"
            + (" ✓ saved" if saved else "")
        )

        if early_cb.should_stop:
            print("  Early stopping triggered.")
            break

    writer.close()
    print(f"\nTraining complete. Best val acc: {ckpt_cb.best_acc:.4f}")
    print(f"Best checkpoint: {ckpt_cb.last_path}")


if __name__ == "__main__":
    main()
