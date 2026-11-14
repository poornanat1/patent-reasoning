"""Draw a seeded sample of decision failures for manual review.

A failure is an example whose predicted set of statutes differs from the gold set (exact
match fails), or that can't be read. Failures are sampled within each gold class in
proportion to the class's share of failures, so every class is represented, and each
record is written with what a reviewer needs: the claim, the gold grounds and examiner
reasoning, and the model's grounds, explanation and citations.

Usage: uv run python scripts/sample_failures.py configs/generation/<variant>.yaml
           --split val --n 40 --out docs/reports/failures/<variant>_val.jsonl
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from patent_reasoning.config import load_config
from patent_reasoning.eval.decision import predicted
from patent_reasoning.schema import Example

SALT = "failures-v1"


def _key(gold: frozenset) -> str:
    return "+".join(sorted(gold))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--split", default="val")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    p = load_config(args.config).params
    rows = {}
    for line in open(Path(p["out_dir"]) / f"{args.split}.jsonl"):
        r = json.loads(line)
        rows[r["example_id"]] = r
    failures = defaultdict(list)
    for line in open(Path(p["splits_dir"]) / f"{args.split}.jsonl"):
        e = Example.model_validate_json(line)
        if e.example_id not in rows:
            continue
        pred = predicted(rows[e.example_id]["text"])
        if pred != e.target.decision:
            failures[_key(e.target.decision)].append((e, pred))

    total = sum(len(v) for v in failures.values())
    rng = random.Random(f"{SALT}:{args.config}:{args.split}")
    picked = []
    for _, items in sorted(failures.items()):
        k = max(1, round(args.n * len(items) / total))
        picked += rng.sample(items, min(k, len(items)))
    rng.shuffle(picked)
    picked = picked[: args.n]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for e, pred in picked:
            r = rows[e.example_id]
            try:
                model = json.loads(r["text"])
            except json.JSONDecodeError:
                model = {"unreadable": r["text"][:500]}
            f.write(json.dumps({
                "example_id": e.example_id,
                "gold": sorted(e.target.decision),
                "predicted": sorted(pred) if pred is not None else None,
                "art_unit": e.provenance.art_unit,
                "claim": e.input.claim.text,
                "parents": [c.text for c in e.input.parent_claims],
                "documents": [d.document_id for d in e.input.prior_art],
                "gold_grounds": [g.model_dump() for g in e.target.grounds],
                "gold_explanation": e.target.explanation,
                "examiner_text": e.provenance.examiner_text,
                "flags": e.provenance.flags,
                "model": model,
                "error": r["error"],
            }) + "\n")  # fmt: skip
    print(f"{total} failures; wrote {len(picked)} to {out}")
    print({k: len(v) for k, v in sorted(failures.items())})


if __name__ == "__main__":
    main()
