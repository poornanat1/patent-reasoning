"""Log a trivial run to MLflow to confirm the environment is set up.

Usage: uv run python scripts/smoke_test.py [configs/smoke.yaml]
"""

import sys

import mlflow

from patent_reasoning.config import load_config
from patent_reasoning.tracking import start_run


def main() -> None:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/smoke.yaml")
    with start_run(cfg) as run:
        mlflow.log_metric("ok", 1.0)
        print(f"Logged run {run.info.run_id} to {cfg.tracking.tracking_uri}")


if __name__ == "__main__":
    main()
