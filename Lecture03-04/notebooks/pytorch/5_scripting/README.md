# CIFAR-10 Classifier

Two-stage transfer learning from a ResNet-18 ImageNet checkpoint to CIFAR-10, packaged as a console-runnable project with a data pipeline, training pipeline, evaluation pipeline, and a FastAPI inference server.

## Project structure

```
cifar10_classifier/
├── src/
│   ├── data/
│   │   ├── dataset.py       # CIFAR-10 DataLoader factory
│   │   └── transforms.py    # ImageNet preprocessing + augmentation
│   ├── models/
│   │   └── resnet.py        # ResNet-18 builder, freeze/unfreeze helpers
│   ├── training/
│   │   ├── trainer.py       # train_one_epoch / evaluate loops
│   │   └── callbacks.py     # ModelCheckpoint, EarlyStopping
│   ├── evaluation/
│   │   └── metrics.py       # accuracy, classification report, confusion matrix
│   ├── serving/
│   │   ├── predictor.py     # inference engine (PIL → top-k predictions)
│   │   └── app.py           # FastAPI server
│   └── utils/
│       ├── logging.py       # TensorBoard writer
│       └── io.py            # checkpoint save/load/prune
├── scripts/
│   ├── prepare_data.py      # data pipeline entry point
│   ├── train.py             # training pipeline entry point
│   ├── evaluate.py          # evaluation pipeline entry point
│   └── serve.py             # deployment pipeline entry point
├── config/
│   └── base.yaml            # all hyperparameters
├── pyproject.toml
└── Makefile
```

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install the package and all dependencies
make install
```

## Running the pipelines

### 1 — Data pipeline

Downloads CIFAR-10 to `data/` and verifies the splits:

```bash
make data
# or explicitly:
python scripts/prepare_data.py --config config/base.yaml
```

### 2 — Training pipeline

Two-stage fine-tuning: stage 1 trains only the FC head, stage 2 unfreezes the full backbone.

```bash
make train
# or with overrides:
python scripts/train.py --config config/base.yaml --stage1-epochs 5 --stage2-epochs 20
```

Checkpoints are saved to `runs/` and TensorBoard logs to `logs/`. Launch TensorBoard with:

```bash
tensorboard --logdir logs/
```

Quick smoke test (1 epoch per stage):

```bash
make smoke
```

### 3 — Evaluation pipeline

Evaluates a checkpoint on the held-out test set and prints a full classification report:

```bash
make evaluate CHECKPOINT=runs/resnet18-two-stage_epoch050_acc0.9312.pt
# save report to disk:
python scripts/evaluate.py --checkpoint runs/best.pt --save-report results/
```

### 4 — Deployment pipeline

Starts a FastAPI server that accepts image uploads and returns top-k predictions:

```bash
make serve CHECKPOINT=runs/resnet18-two-stage_epoch050_acc0.9312.pt
# or with port override:
python scripts/serve.py --checkpoint runs/best.pt --port 8080 --top-k 3
```

**Health check:**
```bash
curl http://localhost:8080/health
```

**Predict:**
```bash
curl -X POST http://localhost:8080/predict -F "file=@cat.jpg"
# {"predictions": [{"class": "cat", "confidence": 0.923}, ...]}
```

Interactive API docs: `http://localhost:8080/docs`

## Configuration

All hyperparameters live in `config/base.yaml`. Any value can be overridden from the CLI:

```bash
python scripts/train.py --config config/base.yaml --batch-size 128 --run-name my-run
```

## Training strategy

| Stage | Trainable params | LR | Rationale |
|-------|-----------------|-----|-----------|
| Stage 1 | FC head only (5,130) | 1e-3 | Fast convergence, no risk of damaging pre-trained features |
| Stage 2 | All (11.2M) | 1e-4 | Full adaptation with a lower LR to preserve learned representations |
