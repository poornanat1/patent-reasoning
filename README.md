# Patent Legal Reasoning

Patent Legal Reasoning compares two ways of improving an open 7B–8B model at legal reasoning over USPTO patent examination records:

- **Fine-tuning:** QLoRA SFT, then DPO
- **Prompt optimization:** DSPy + GEPA

The full design is in [docs/SPEC.md](docs/SPEC.md).

> Status: early setup. Data pipeline in progress.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # core data + tracking deps
uv sync --extra train        # GPU training stack (Linux + CUDA)
uv sync --all-extras         # everything
```

Verify the environment by running the smoke test and the test suite:

```bash
uv run python scripts/smoke_test.py
uv run pytest
```

## Experiment tracking

Runs are logged to MLflow, stored in a local SQLite database (`mlflow.db`). To browse them:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Every run records its config, seed and git commit.

## Layout

```
configs/   YAML experiment configs
data/      raw / interim / processed data (not committed)
docs/      specification and reports
scripts/   entry points
src/       patent_reasoning package
tests/     unit tests
```
