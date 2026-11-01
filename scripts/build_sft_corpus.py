"""Build the initial SFT corpus from claim records and the evidence dataset.

Writes to the config's out_dir:

- examples.jsonl: schema-valid ``Example``s (see patent_reasoning.data.sft for what qualifies)
- excluded.jsonl: every other claim record with the reason it was left out

The summary goes to the config's report path and MLflow.

Usage: uv run python scripts/build_sft_corpus.py [configs/sft_corpus.yaml]
"""

import argparse
import json
from pathlib import Path

import mlflow

from patent_reasoning.config import load_config
from patent_reasoning.data.sft import build_example, summarize_outcomes
from patent_reasoning.tracking import start_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/sft_corpus.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    p = cfg.params
    out_dir = Path(p["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    records = {r["example_id"]: r for r in map(json.loads, open(p["claim_records"]))}
    documents = {d["document_id"]: d for d in map(json.loads, open(p["documents"]))}
    retrieval = {x["example_id"]: x for x in map(json.loads, open(p["retrieval"]))}

    with start_run(cfg):
        outcomes = [
            build_example(r, documents, retrieval[eid]["passage_ids"],
                          retrieval[eid]["condition"], p["max_target_tokens"])
            for eid, r in records.items()
        ]  # fmt: skip
        with open(out_dir / "examples.jsonl", "w") as f:
            f.writelines(o.example.model_dump_json() + "\n" for o in outcomes if o.example)
        with open(out_dir / "excluded.jsonl", "w") as f:
            f.writelines(
                json.dumps({"example_id": o.example_id, "reason": o.reason, "detail": o.detail,
                            "decision": records[o.example_id]["decision"],
                            "quality": records[o.example_id]["quality"]}) + "\n"
                for o in outcomes if not o.example
            )  # fmt: skip
        summary = summarize_outcomes(outcomes, records)
        Path(p["report"]).write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, int | float)})
        mlflow.log_dict(summary, "summary.json")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
