"""Build DPO preference pairs from the train and validation splits.

Writes {split}.jsonl (one pair per line: input, chosen, rejected) to the config's out_dir,
and the summary to the config's report path and MLflow. False-allowance pairs are capped
at the number of false-rejection pairs (``balance_decision_pairs``). See
patent_reasoning.data.preferences for the pair types.

Usage: uv run python scripts/build_preferences.py [configs/preferences.yaml]
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import mlflow

from patent_reasoning.config import load_config
from patent_reasoning.data.preferences import PAIR_TYPES, Pair, make_pairs
from patent_reasoning.data.splits import EVAL_SPLITS
from patent_reasoning.schema import Example
from patent_reasoning.tracking import start_run


def balance(pairs: list[Pair]) -> list[Pair]:
    """Keep no more false-allowance pairs (which push toward rejecting) than false-rejection
    pairs (which push toward allowing), so DPO doesn't shift the decision prior."""
    limit = sum(pair.type == "false_rejection" for pair in pairs)
    allowance = sorted(
        (pair for pair in pairs if pair.type == "false_allowance"),
        key=lambda pair: hashlib.sha256(pair.pair_id.encode()).hexdigest(),
    )
    keep = {pair.pair_id for pair in allowance[:limit]}
    return [pair for pair in pairs if pair.type != "false_allowance" or pair.pair_id in keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/preferences.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    p = cfg.params
    assert not set(p["splits"]) & (set(EVAL_SPLITS) - {"val"}), "test sets never feed DPO"
    out_dir = Path(p["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    limitations = {
        r["example_id"]: r["limitations"] for r in map(json.loads, open(p["claim_records"]))
    }

    with start_run(cfg):
        summary = {"pair_types": list(PAIR_TYPES), "splits": {}}
        for split in p["splits"]:
            examples = [Example.model_validate_json(line)
                        for line in open(Path(p["splits_dir"]) / f"{split}.jsonl")]  # fmt: skip
            pairs = [pair for e in examples
                     for pair in make_pairs(e, split, limitations[e.example_id])]  # fmt: skip
            if p["balance_decision_pairs"]:
                pairs = balance(pairs)
            with open(out_dir / f"{split}.jsonl", "w") as f:
                f.writelines(json.dumps(pair.to_dict()) + "\n" for pair in pairs)
            by_type = Counter(pair.type for pair in pairs)
            summary["splits"][split] = {
                "examples": len(examples),
                "examples_with_pairs": len({pair.example_id for pair in pairs}),
                "pairs": len(pairs),
                "pairs_by_type": {t: by_type[t] for t in PAIR_TYPES},
                "chosen_decision": dict(Counter(
                    "+".join(sorted(pair.example.target.decision)) for pair in pairs
                ).most_common()),
                "rejected_decision": dict(Counter(
                    "+".join(sorted(pair.rejected.decision)) for pair in pairs
                ).most_common()),
            }  # fmt: skip
        Path(p["report"]).write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_metrics({f"{s}_pairs": v["pairs"] for s, v in summary["splits"].items()})
        mlflow.log_dict(summary, "summary.json")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
