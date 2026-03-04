# predict.py
# ───────────────────────────────────────────────────────────────
# Standalone inference script.
# Loads a saved checkpoint and runs predictions on the test set.
#
# Run with:
#   python predict.py --checkpoint checkpoints/mnist-epoch=05-val_acc=0.975.ckpt
# ───────────────────────────────────────────────────────────────

import argparse

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data  import MNISTDataModule
from model import MNISTClassifier


def main(checkpoint_path: str, num_samples: int = 16):
    # ── Load model from checkpoint ─────────────────────────────
    # load_from_checkpoint works because we called save_hyperparameters()
    # in MNISTClassifier — no need to pass constructor args manually.
    model = MNISTClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    print(f"Loaded model from: {checkpoint_path}")
    print(f"Hyperparameters  : {model.hparams}")

    # ── Load test data ─────────────────────────────────────────
    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup(stage="test")

    batch        = next(iter(dm.test_dataloader()))
    images, labels = batch

    # ── Inference ──────────────────────────────────────────────
    with torch.no_grad():
        logits = model(images[:num_samples])
        probs  = F.softmax(logits, dim=1)
        preds  = torch.argmax(probs, dim=1)

    # ── Accuracy on this batch ─────────────────────────────────
    acc = (preds == labels[:num_samples]).float().mean()
    print(f"\nAccuracy on {num_samples} samples: {acc:.2%}")

    # ── Plot ───────────────────────────────────────────────────
    cols = 8
    rows = num_samples // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 2))

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap="gray")
        pred_label = preds[i].item()
        true_label = labels[i].item()
        confidence = probs[i, pred_label].item()
        color = "green" if pred_label == true_label else "red"
        ax.set_title(f"P:{pred_label} T:{true_label}\n{confidence:.0%}", color=color, fontsize=8)
        ax.axis("off")

    plt.suptitle("Predictions  (green=correct, red=wrong)", y=1.01)
    plt.tight_layout()
    plt.savefig("predictions.png", bbox_inches="tight")
    plt.show()
    print("Plot saved to predictions.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .ckpt file",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of test samples to visualize",
    )
    args = parser.parse_args()
    main(args.checkpoint, args.num_samples)
