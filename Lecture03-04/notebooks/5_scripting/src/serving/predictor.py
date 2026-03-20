"""Single-image and batch inference against a loaded model."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from src.data.transforms import get_inference_transform
from src.data.dataset import CLASSES


class Predictor:
    """Wraps a trained model for production inference.

    Parameters
    ----------
    model      : trained nn.Module already on the correct device
    device     : torch device
    top_k      : number of top-k predictions to return
    """

    def __init__(self, model: nn.Module, device: torch.device, top_k: int = 5):
        self.model   = model.eval()
        self.device  = device
        self.top_k   = top_k
        self.transform = get_inference_transform()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, top_k: int = 5) -> "Predictor":
        """Convenience constructor: loads model weights from a .pt checkpoint."""
        from src.models.resnet import build_model
        from src.utils.io import load_checkpoint

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = build_model(num_classes=len(CLASSES))
        load_checkpoint(str(checkpoint_path), model)
        model.to(device)
        return cls(model, device, top_k=top_k)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> list[dict]:
        """Return top-k predictions for a PIL Image.

        Returns
        -------
        list of dicts, each with keys ``class``, ``confidence``.
        """
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

        values, indices = probs.topk(min(self.top_k, len(CLASSES)))
        return [
            {"class": CLASSES[idx.item()], "confidence": round(val.item(), 6)}
            for val, idx in zip(values, indices)
        ]

    @torch.inference_mode()
    def predict_batch(self, images: list[Image.Image]) -> list[list[dict]]:
        """Predict a batch of PIL images in a single forward pass."""
        tensors = torch.stack([self.transform(img) for img in images]).to(self.device)
        logits  = self.model(tensors)
        probs   = torch.softmax(logits, dim=1)

        results = []
        for row in probs:
            values, indices = row.topk(min(self.top_k, len(CLASSES)))
            results.append(
                [{"class": CLASSES[idx.item()], "confidence": round(val.item(), 6)}
                 for val, idx in zip(values, indices)]
            )
        return results
