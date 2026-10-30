"""Experiment configuration loaded from YAML.

Every run is described by one YAML file under ``configs/``. The resolved config is
logged to MLflow so a run can be reproduced from its tracked parameters alone.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class TrackingConfig(BaseModel):
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment: str = "patent-reasoning"


class ExperimentConfig(BaseModel):
    name: str
    seed: int = 42
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    params: dict[str, Any] = Field(default_factory=dict)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        return ExperimentConfig.model_validate(yaml.safe_load(f))


def set_seed(seed: int) -> None:
    """Seed every RNG in use. numpy/torch are seeded only if installed."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
