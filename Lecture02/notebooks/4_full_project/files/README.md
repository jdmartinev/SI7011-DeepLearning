# MNIST Classifier — PyTorch Lightning Project

A clean, production-style project layout for PyTorch Lightning.  
Compare this with the tutorial notebook to see how the same concepts translate into real project structure.

---

## Project Structure

```
mnist_project/
│
├── configs/
│   └── config.yaml      ← All hyperparameters live here
│
├── data.py              ← LightningDataModule (download, split, DataLoaders)
├── model.py             ← LightningModule (architecture, loss, optimizer, scheduler)
├── train.py             ← Entry point: builds everything and calls trainer.fit()
├── predict.py           ← Loads a checkpoint and runs inference
│
├── requirements.txt
└── README.md
```

### Why this separation?

| File | Single responsibility |
|---|---|
| `config.yaml` | All numbers and settings — change an experiment without touching code |
| `data.py` | Everything about data — reusable across different models |
| `model.py` | Everything about the model — testable in isolation |
| `train.py` | Wires everything together — the only file that imports all others |
| `predict.py` | Inference only — no training code, no data splitting |

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train with default config
python train.py

# 3. Train with a custom config
python train.py --config configs/config.yaml

# 4. Run inference on a saved checkpoint
python predict.py --checkpoint checkpoints/mnist-epoch=05-val_acc=0.975.ckpt
```

---

## Changing Hyperparameters

Edit `configs/config.yaml` — no code changes needed:

```yaml
model:
  lr: 0.0005        # ← change learning rate here
  hidden_size: 512  # ← change model size here

trainer:
  max_epochs: 50    # ← train longer here
```

---

## Outputs

| Path | Contents |
|---|---|
| `logs/mnist_classifier/` | CSVLogger metrics — open `metrics.csv` to plot curves |
| `checkpoints/` | Best model checkpoint (`.ckpt` file) |
| `predictions.png` | Inference visualization from `predict.py` |
