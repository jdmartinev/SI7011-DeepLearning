"""ResNet-18 adapted for CIFAR-10 with two-stage transfer learning."""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(num_classes: int = 10, weights: str = "IMAGENET1K_V1") -> nn.Module:
    """Load a pre-trained ResNet-18 and replace the final FC for CIFAR-10.

    All backbone layers start frozen (stage 1 setup). Call
    ``unfreeze_backbone`` to transition to stage 2.
    """
    model = resnet18(weights=ResNet18_Weights[weights])

    # Freeze all pre-trained weights
    for param in model.parameters():
        param.requires_grad = False

    # Replace the head — new layers have requires_grad=True by default
    in_features = model.fc.in_features          # 512 for resnet18
    model.fc = nn.Linear(in_features, num_classes)

    return model


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters for full fine-tuning (stage 2)."""
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> dict[str, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
