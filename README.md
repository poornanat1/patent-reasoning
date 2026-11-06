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

Next, build one record per claim from each application's first non-final action, and the evidence each claim's input will show:

```bash
scripts/build_dataset.sh                      # every step below through the manifest; resumes from the ODP cache
uv run python scripts/build_claim_records.py   # -> data/interim/claim_records/claims.jsonl
uv run python scripts/build_evidence.py        # -> data/interim/evidence/{documents,retrieval}.jsonl
uv run python scripts/build_sft_corpus.py      # -> data/interim/sft_corpus/{examples,excluded}.jsonl
uv run python scripts/build_splits.py          # -> data/processed/splits/{train,val,test,ood_test}.jsonl
uv run python scripts/build_preferences.py     # -> data/processed/preferences/{train,val}.jsonl
uv run python scripts/audit_dataset.py sheet   # -> data/interim/audit/batch-*.md (manual review sheets)
uv run python scripts/audit_dataset.py summarize  # -> docs/reports/dataset_audit_summary.json
uv run python scripts/build_manifest.py        # -> data/processed/manifest.json (dataset v1)
```

Findings so far: [audit](docs/reports/oard_audit_findings.md), [feasibility](docs/reports/feasibility.md), [claim-level dataset and evidence](docs/reports/claim_level_dataset.md), [SFT corpus](docs/reports/sft_corpus.md), [splits and preference pairs](docs/reports/splits_and_preferences.md), [dataset v1 audit](docs/reports/dataset_v1.md).

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
