# Reproducing

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # core data + tracking deps
uv sync --extra train        # GPU training stack (Linux + CUDA)
uv sync --all-extras         # everything
```

To check the environment, run the smoke test and the test suite:

```bash
uv run python scripts/smoke_test.py
uv run pytest
```

## Data

The pipeline starts from the [USPTO Office Action Research Dataset](https://www.uspto.gov/ip-policy/economic-research/research-datasets/office-action-research-dataset-patents) (2017 release). It holds about 4.4M office actions from 2008 to mid-2017, with rejection and citation tables. The USPTO Open Data Portal serves the files, and downloading them needs a free API key from [data.uspto.gov](https://data.uspto.gov).

```bash
export USPTO_API_KEY=...
uv run python scripts/download_oard.py      # ~800 MB of zips into data/raw/oard
uv run python scripts/sample_oard.py        # extract + 1% application sample -> data/interim
uv run python scripts/audit_oard.py         # schema/quality report -> docs/results
uv run python scripts/audit_oard.py --sample
```

If you don't have a key, download `office_actions.csv.zip`, `rejections.csv.zip` and `citations.csv.zip` in a browser from the [PTOFFACT dataset page](https://data.uspto.gov/bulkdata/datasets/ptoffact). Put them in `data/raw/oard/` and start from the sample step.

The sample takes whole applications, chosen by a salted SHA-256 of `app_id`. A rerun gives the same sample whatever the row order, and every office action, rejection and citation of a chosen application stays together. That keeps the later family-level splits clean. The sample settings are in [configs/oard_sample.yaml](../configs/oard_sample.yaml).

The next step joins each sampled office action with its text, its claims and the cited prior art from the Open Data Portal file wrapper. It uses the same API key, read from the environment or from `.env`. Responses are cached under `data/raw/odp/`.

```bash
uv run python scripts/build_joined_sample.py   # -> data/interim/joined_sample/records.jsonl
uv run python scripts/inspection_sheet.py      # Markdown sheet for manual review
```

Then build one record per claim from each application's first non-final action, along with the evidence that claim's input will show. `scripts/build_dataset.sh` runs every step below through the manifest, and resumes from the ODP cache:

```bash
scripts/build_dataset.sh
uv run python scripts/build_claim_records.py   # -> data/interim/claim_records/claims.jsonl
uv run python scripts/build_evidence.py        # -> data/interim/evidence/{documents,retrieval}.jsonl
uv run python scripts/build_sft_corpus.py      # -> data/interim/sft_corpus/{examples,excluded}.jsonl
uv run python scripts/build_splits.py          # -> data/processed/splits/{train,val,test,ood_test}.jsonl
uv run python scripts/build_preferences.py     # -> data/processed/preferences/{train,val}.jsonl
uv run python scripts/audit_dataset.py sheet   # -> data/interim/audit/batch-*.md (manual review sheets)
uv run python scripts/audit_dataset.py summarize  # -> docs/audits/dataset_audit_summary.json
uv run python scripts/audit_dataset.py summarize configs/dataset_audit_v2.yaml  # second audit, with error rates
uv run python scripts/build_manifest.py        # -> data/processed/manifest.json (dataset v1)
```

[data/README.md](../data/README.md) describes every directory the pipeline writes.

## Generation on a laptop

This needs the `infer` extra (MLX, Apple silicon). The first run downloads Qwen3-8B (4.6 GB).

```bash
uv sync --extra infer
uv run python scripts/measure_prompt_tokens.py # -> docs/results/prompt_tokens.json
uv run python scripts/generate.py configs/generation/base_zero_shot.yaml --split val  # resumable
scripts/run_baselines.sh                      # zero-shot and few-shot on every evaluation split
uv run python scripts/score.py configs/generation/base_zero_shot.yaml --split val  # -> <out_dir>/val.scores.json
```

## The comparison on a GPU

Every reported comparison runs on a CUDA machine with one A100 80 GB. Install the pinned packages there, then generate with vLLM and train with QLoRA:

```bash
uv venv -p 3.12 venv && . venv/bin/activate && uv pip install -r requirements/gpu.txt -e .
python scripts/generate.py configs/generation/gpu_base_zero_shot.yaml --split val   # batched, bf16
python scripts/train_sft.py configs/training/sft.yaml --max-steps 20 --out-dir checkpoints/sft-smoke
python scripts/train_sft.py configs/training/sft.yaml                               # -> checkpoints/sft/final
python scripts/generate.py configs/generation/gpu_sft.yaml --split val               # the adapter on bf16
```

## Experiment tracking

Runs are logged to MLflow, in a local SQLite database (`mlflow.db`). Each run records its config, seed and git commit. To browse them:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
