#!/usr/bin/env python
"""Data pipeline: download CIFAR-10 and verify integrity.

Usage
-----
    python scripts/prepare_data.py
    python scripts/prepare_data.py --data-dir /tmp/data --config config/base.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and verify CIFAR-10")
    p.add_argument("--data-dir", default=None, help="Override data.root from config")
    p.add_argument("--config",   default="config/base.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = args.data_dir or cfg["data"]["root"]
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    print(f"Downloading CIFAR-10 to '{data_dir}' ...")
    import torchvision
    train = torchvision.datasets.CIFAR10(root=data_dir, train=True,  download=True)
    test  = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)

    print(f"  Train samples : {len(train):,}")
    print(f"  Test  samples : {len(test):,}")
    print(f"  Classes       : {train.classes}")
    print("Done.")


if __name__ == "__main__":
    main()
