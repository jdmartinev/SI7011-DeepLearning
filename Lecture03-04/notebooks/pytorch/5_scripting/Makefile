.PHONY: install data train evaluate serve clean lint test

# ── Installation ───────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

# ── Pipelines ──────────────────────────────────────────────────────────────────
data:
	python scripts/prepare_data.py --config config/base.yaml

train:
	python scripts/train.py --config config/base.yaml

evaluate:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Usage: make evaluate CHECKPOINT=runs/<name>.pt"; exit 1; \
	fi
	python scripts/evaluate.py --checkpoint $(CHECKPOINT) --config config/base.yaml

serve:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Usage: make serve CHECKPOINT=runs/<name>.pt"; exit 1; \
	fi
	python scripts/serve.py --checkpoint $(CHECKPOINT) --config config/base.yaml

# ── Quick overrides ────────────────────────────────────────────────────────────
# Fast smoke-test: 1 epoch per stage, small batch
smoke:
	python scripts/train.py \
		--config config/base.yaml \
		--stage1-epochs 1 \
		--stage2-epochs 1 \
		--batch-size 128 \
		--run-name smoke-test

# ── Dev tools ──────────────────────────────────────────────────────────────────
lint:
	ruff check src/ scripts/
	ruff format --check src/ scripts/

format:
	ruff format src/ scripts/
	ruff check --fix src/ scripts/

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# ── Housekeeping ───────────────────────────────────────────────────────────────
clean:
	rm -rf runs/ logs/ data/ __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
