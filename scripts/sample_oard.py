"""Extract the OARD zips and write an application-level sample to Parquet.

Logs the sample counts and a manifest (source hashes, parameters) to MLflow.

Usage: uv run python scripts/sample_oard.py [configs/oard_sample.yaml]
"""

import json
import sys
from pathlib import Path

import mlflow

from patent_reasoning.config import load_config
from patent_reasoning.data.oard import (
    TABLES,
    build_sample,
    connect,
    extract_tables,
    sha256,
    zip_name,
)
from patent_reasoning.tracking import start_run


def main() -> None:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/oard_sample.yaml")
    p = cfg.params
    zip_dir, out_dir = Path(p["zip_dir"]), Path(p["out_dir"])

    with start_run(cfg):
        csvs = extract_tables(zip_dir, Path(p["csv_dir"]))
        counts = build_sample(
            connect(csvs), out_dir, p["sample_rate"], p["year_min"], p["year_max"], p["salt"]
        )
        manifest = {
            "source": "USPTO Office Action Research Dataset, 2017 release (ODP PTOFFACT)",
            "sources": {t: {"file": zip_name(t), "sha256": sha256(zip_dir / zip_name(t))}
                        for t in TABLES},
            "params": p,
            "counts": counts,
        }  # fmt: skip
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        mlflow.log_metrics({f"rows_{k}": v for k, v in counts.items()})
        mlflow.log_dict(manifest, "manifest.json")
        print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
