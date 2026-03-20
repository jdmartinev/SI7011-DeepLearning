"""Thin wrapper around SummaryWriter with run-name support."""
from __future__ import annotations

import os
from torch.utils.tensorboard import SummaryWriter


def get_writer(log_dir: str, run_name: str) -> SummaryWriter:
    path = os.path.join(log_dir, run_name)
    os.makedirs(path, exist_ok=True)
    return SummaryWriter(log_dir=path)
