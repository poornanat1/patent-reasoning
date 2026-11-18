"""Per-class decision thresholds over sampled decisions (scripts/sample_decisions.py).

An example's score for a statute is the share of its samples that include it. The rule:
predict every statute whose score reaches its threshold, and "no rejection" if none does.
Thresholds are fitted to maximise macro-F1 by coordinate ascent over the possible shares,
starting from 0.5 (a majority vote).

``fit`` (on validation) reports two estimates, both against greedy decoding on the same
examples:
- cross-validated: thresholds fitted on one half of the applications and applied to the
  other, both ways round, so the score isn't flattered by fitting and scoring on the same
  examples;
- in-sample: thresholds fitted on all of validation (these are the ones saved).
``apply`` uses saved thresholds on another split and changes nothing.

Usage: uv run python scripts/tune_thresholds.py fit configs/generation/gpu_sft.yaml
           --split val --out <file.json>
       uv run python scripts/tune_thresholds.py apply configs/generation/gpu_sft.yaml
           --split test --thresholds <file.json>
"""

import argparse
import hashlib
import json
from pathlib import Path

from patent_reasoning.config import load_config
from patent_reasoning.eval.decision import (
    STATUTES,
    bootstrap,
    macro_f1,
    metrics,
    paired_bootstrap,
    predicted,
)
from patent_reasoning.schema import NO_REJECTION, Example

ORDER = ["102", "103", "112", "101"]
assert set(ORDER) == STATUTES
SALT = "thresholds-v1"


def load(config: str, split: str) -> list[dict]:
    p = load_config(config).params
    out = Path(p["out_dir"])
    samples = {(r := json.loads(line))["example_id"]: r
               for line in open(out / f"{split}.samples.jsonl")}  # fmt: skip
    greedy = {(r := json.loads(line))["example_id"]: r["text"]
              for line in open(out / f"{split}.jsonl")}  # fmt: skip
    rows = []
    for line in open(Path(p["splits_dir"]) / f"{split}.jsonl"):
        e = Example.model_validate_json(line)
        if e.example_id not in samples or e.example_id not in greedy:
            continue
        ds = samples[e.example_id]["decisions"]
        rows.append({
            "app": e.provenance.application_id,
            "gold": e.target.decision,
            "greedy": predicted(greedy[e.example_id]),
            "score": {c: sum(d is not None and c in d for d in ds) / len(ds) for c in ORDER},
            "n": len(ds),
        })  # fmt: skip
    return rows


def decide(row: dict, t: dict[str, float]) -> frozenset:
    s = frozenset(c for c in ORDER if row["score"][c] >= t[c])
    return s or frozenset({NO_REJECTION})


def fit(rows: list[dict], rounds: int = 4) -> dict[str, float]:
    n = rows[0]["n"]
    grid = [k / n for k in range(1, n + 1)] + [1.01]  # 1.01: never predict the class
    gold = [r["gold"] for r in rows]
    t = {c: 0.5 for c in ORDER}
    for _ in range(rounds):
        for c in ORDER:
            t[c] = max(grid, key=lambda v: (macro_f1(gold, [decide(r, t | {c: v}) for r in rows]),
                                            -abs(v - 0.5)))  # fmt: skip
    return t


def fold(row: dict) -> int:
    return int(hashlib.sha256(f"{SALT}:{row['app']}".encode()).hexdigest(), 16) % 2


def report(rows: list[dict], pred: list, label: str) -> dict:
    gold = [r["gold"] for r in rows]
    apps = [r["app"] for r in rows]
    m = metrics(gold, pred)
    return {
        "label": label,
        "macro_f1": m["macro_f1"],
        "macro_f1_ci95": bootstrap(gold, pred, apps),
        "per_class_f1": {c: v["f1"] for c, v in m["per_class"].items()},
        "vs_greedy": paired_bootstrap(gold, [r["greedy"] for r in rows], pred, apps),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["fit", "apply"])
    parser.add_argument("config")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", help="fit: where to save the thresholds")
    parser.add_argument("--thresholds", help="apply: saved thresholds")
    args = parser.parse_args()
    rows = load(args.config, args.split)
    greedy = [r["greedy"] for r in rows]
    result = {"split": args.split, "examples": len(rows),
              "greedy_macro_f1": macro_f1([r["gold"] for r in rows], greedy)}  # fmt: skip
    if args.mode == "fit":
        majority = report(rows, [decide(r, dict.fromkeys(ORDER, 0.5)) for r in rows],
                          "majority vote (all thresholds 0.5)")  # fmt: skip
        cv_pred = [None] * len(rows)
        cv_thresholds = []
        for k in (0, 1):
            train = [r for r in rows if fold(r) != k]
            t = fit(train)
            cv_thresholds.append(t)
            for i, r in enumerate(rows):
                if fold(r) == k:
                    cv_pred[i] = decide(r, t)
        t = fit(rows)
        result |= {
            "majority_vote": majority,
            "cross_validated": report(rows, cv_pred, "thresholds fitted on the other half")
            | {"fold_thresholds": cv_thresholds},
            "in_sample": report(rows, [decide(r, t) for r in rows], "fitted on all of it"),
            "thresholds": t,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"config": args.config, "fitted_on": args.split, "thresholds": t},
                  open(args.out, "w"), indent=2)  # fmt: skip
    else:
        t = json.load(open(args.thresholds))["thresholds"]
        result |= {"applied": report(rows, [decide(r, t) for r in rows], "saved thresholds"),
                   "thresholds": t}  # fmt: skip
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
