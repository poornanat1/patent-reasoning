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

## Data

The pipeline starts from the [USPTO Office Action Research Dataset](https://www.uspto.gov/ip-policy/economic-research/research-datasets/office-action-research-dataset-patents) (2017 release): about 4.4M office actions from 2008 to mid-2017, with rejection and citation tables. The USPTO Open Data Portal serves the files. Downloading them needs a free API key from [data.uspto.gov](https://data.uspto.gov).

```bash
export USPTO_API_KEY=...
uv run python scripts/download_oard.py      # ~800 MB of zips into data/raw/oard
uv run python scripts/sample_oard.py        # extract + 1% application sample -> data/interim
uv run python scripts/audit_oard.py         # schema/quality report -> docs/reports
uv run python scripts/audit_oard.py --sample
```

Without a key, download `office_actions.csv.zip`, `rejections.csv.zip` and `citations.csv.zip` from the [PTOFFACT dataset page](https://data.uspto.gov/bulkdata/datasets/ptoffact) in a browser, put them in `data/raw/oard/`, and start from the sample step.

Then join each sampled office action with its text, claims and cited prior art from the Open Data Portal file wrapper. The same API key is used (from the environment or `.env`). Responses are cached under `data/raw/odp/`.

```bash
uv run python scripts/build_joined_sample.py   # -> data/interim/joined_sample/records.jsonl
uv run python scripts/inspection_sheet.py      # Markdown sheet for manual review
```

Findings so far: [audit](docs/reports/oard_audit_findings.md), [feasibility](docs/reports/feasibility.md).

The sample takes whole applications, chosen by a salted SHA-256 of `app_id`. Rerunning it gives the same sample regardless of row order, and every office action, rejection and citation for a chosen application stays together. That keeps later family-level splits clean. Sample settings live in [configs/oard_sample.yaml](configs/oard_sample.yaml).

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
