"""MLflow helpers that attach config, seed, and code version to every run."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager

import mlflow

from patent_reasoning.config import ExperimentConfig, set_seed


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return bool(out.strip())


@contextmanager
def start_run(cfg: ExperimentConfig, run_name: str | None = None) -> Iterator[mlflow.ActiveRun]:
    """Seed RNGs and open an MLflow run with the full config logged."""
    set_seed(cfg.seed)
    mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
    mlflow.set_experiment(cfg.tracking.experiment)
    with mlflow.start_run(run_name=run_name or cfg.name) as run:
        mlflow.set_tags({"git_commit": git_commit(), "git_dirty": str(git_dirty())})
        mlflow.log_params({"seed": cfg.seed, **cfg.params})
        mlflow.log_dict(cfg.model_dump(), "config.json")
        yield run
