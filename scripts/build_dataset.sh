#!/usr/bin/env bash
# Rebuild the dataset from the OARD sample: claim records through the v1 manifest.
# Every Open Data Portal response is cached under data/raw/odp, so a stopped run resumes
# where it left off when started again.
#
# Usage: scripts/build_dataset.sh  (log: data/interim/build_dataset.log when run detached)
set -euo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1
for step in build_claim_records build_evidence build_sft_corpus build_splits build_preferences; do
  echo "== $step $(date '+%F %T')"
  uv run python "scripts/$step.py"
done
echo "== audit summary $(date '+%F %T')"
uv run python scripts/audit_dataset.py summarize
echo "== manifest $(date '+%F %T')"
uv run python scripts/build_manifest.py
echo "== done $(date '+%F %T')"
