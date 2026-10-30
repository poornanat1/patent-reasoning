import random

import mlflow

from patent_reasoning.config import ExperimentConfig, load_config, set_seed
from patent_reasoning.tracking import start_run


def test_load_smoke_config():
    cfg = load_config("configs/smoke.yaml")
    assert cfg.name == "smoke"
    assert cfg.seed == 42


def test_set_seed_is_deterministic():
    set_seed(7)
    a = random.random()
    set_seed(7)
    assert random.random() == a


def test_start_run_logs_config(tmp_path):
    cfg = ExperimentConfig.model_validate(
        {
            "name": "test",
            "tracking": {"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db", "experiment": "t"},
            "params": {"lr": 2e-4},
        }
    )
    with start_run(cfg) as run:
        run_id = run.info.run_id
    logged = mlflow.get_run(run_id)
    assert logged.data.params == {"seed": "42", "lr": "0.0002"}
    assert "git_commit" in logged.data.tags
