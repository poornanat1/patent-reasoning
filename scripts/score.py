"""Score one variant's generations on one split: decisions, citations, and slices.

Reads ``<out_dir>/<split>.jsonl`` from the variant's generation config, joins it to the
split, and writes ``<out_dir>/<split>.scores.json`` and MLflow metrics:

- ``decision``: per-class precision/recall/F1, macro-F1 with a 95% interval that resamples
  applications, false-rejection and false-allowance rates
- ``validity``: outputs that parse into a ``ModelOutput``, and outputs cut off at the limit
- ``faithfulness``: citation existence and support, unsupported assertions, agreement with
  the examiner's pin cites (see patent_reasoning.eval.faithfulness)
- ``slices``: decision metrics by technology center and by mail year

With ``--compare`` (another generation config), also a paired bootstrap of the macro-F1
difference on the examples both variants generated.

Usage: uv run python scripts/score.py configs/generation/<variant>.yaml --split val
           [--compare configs/generation/<other>.yaml]
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import mlflow

from patent_reasoning.config import load_config
from patent_reasoning.eval.decision import bootstrap, metrics, paired_bootstrap, predicted
from patent_reasoning.eval.faithfulness import check, score
from patent_reasoning.schema import Example
from patent_reasoning.tracking import start_run

MIN_SLICE = 30  # smaller slices are listed with their size but no metrics


def load(config: str, split: str) -> tuple[dict, dict[str, dict]]:
    p = load_config(config).params
    rows = {}
    for line in open(Path(p["out_dir"]) / f"{split}.jsonl"):
        r = json.loads(line)
        rows[r["example_id"]] = r
    return p, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--split", default="val")
    parser.add_argument("--compare", help="generation config of the baseline to compare with")
    args = parser.parse_args()
    cfg = load_config(args.config)
    p, rows = load(args.config, args.split)
    examples = [Example.model_validate_json(line)
                for line in open(Path(p["splits_dir"]) / f"{args.split}.jsonl")]  # fmt: skip
    scored = [e for e in examples if e.example_id in rows]
    gold = [e.target.decision for e in scored]
    pred = [predicted(rows[e.example_id]["text"]) for e in scored]
    apps = [e.provenance.application_id for e in scored]

    decision = metrics(gold, pred)
    decision["macro_f1_ci95"] = bootstrap(gold, pred, apps)
    counts = Counter()
    for e in scored:
        counts += check(e, rows[e.example_id]["text"])
    slices: dict[str, dict] = defaultdict(dict)
    for name, key in (("technology_center", lambda e: e.provenance.art_unit[:2] + "00"),
                      ("mail_year", lambda e: e.provenance.mail_date[:4])):  # fmt: skip
        groups = defaultdict(list)
        for i, e in enumerate(scored):
            groups[key(e)].append(i)
        for value, idx in sorted(groups.items()):
            m = {"examples": len(idx)}
            if len(idx) >= MIN_SLICE:
                full = metrics([gold[i] for i in idx], [pred[i] for i in idx])
                m |= {k: full[k] for k in ("macro_f1", "exact_match", "false_rejection_rate",
                                           "false_allowance_rate")}  # fmt: skip
            slices[name][value] = m

    gen = [rows[e.example_id] for e in scored]
    report = {
        "variant": cfg.name,
        "split": args.split,
        "examples_in_split": len(examples),
        "examples_scored": len(scored),
        "decision": decision,
        "validity": {
            "valid_rate": round(sum(r["error"] is None for r in gen) / len(gen), 4),
            "truncated_rate": round(sum(r["finish_reason"] == "length" for r in gen) / len(gen), 4),
            "errors": dict(Counter((r["error"] or "").split(":")[0] for r in gen if r["error"])),
        },
        "faithfulness": score(counts),
        "slices": slices,
        "cost": {
            "seconds_total": round(sum(r["seconds"] for r in gen), 1),
            "seconds_per_example": round(sum(r["seconds"] for r in gen) / len(gen), 2),
            "prompt_tokens": sum(r["prompt_tokens"] for r in gen),
            "completion_tokens": sum(r["completion_tokens"] for r in gen),
        },
    }
    if args.compare:
        base_cfg = load_config(args.compare)
        _, base = load(args.compare, args.split)
        both = [i for i, e in enumerate(scored) if e.example_id in base]
        report["compare"] = {
            "baseline": base_cfg.name,
            "examples": len(both),
            "macro_f1": paired_bootstrap(
                [gold[i] for i in both],
                [predicted(base[scored[i].example_id]["text"]) for i in both],
                [pred[i] for i in both],
                [apps[i] for i in both],
            ),
        }

    out = Path(p["out_dir"]) / f"{args.split}.scores.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    with start_run(cfg, run_name=f"{cfg.name}-{args.split}-score"):
        mlflow.set_tags({"run_split": args.split, "stage": "score"})
        f = report["faithfulness"]
        mlflow.log_metrics({
            k: v for k, v in {
                "macro_f1": decision["macro_f1"], "exact_match": decision["exact_match"],
                "false_rejection_rate": decision["false_rejection_rate"],
                "false_allowance_rate": decision["false_allowance_rate"],
                "valid_rate": report["validity"]["valid_rate"],
                "citation_precision": f["citation_precision"],
                "unsupported_assertion_rate": f["unsupported_assertion_rate"],
                "examiner_pin_recall": f["examiner_pin_recall"],
            }.items() if v is not None
        })  # fmt: skip
        mlflow.log_dict(report, f"{args.split}.scores.json")
    summary = {k: report[k] for k in ("examples_scored", "validity")}
    summary["decision"] = {k: decision[k] for k in ("macro_f1", "macro_f1_ci95", "exact_match",
                           "false_rejection_rate", "false_allowance_rate")}  # fmt: skip
    summary["faithfulness"] = {k: v for k, v in report["faithfulness"].items() if k != "counts"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
