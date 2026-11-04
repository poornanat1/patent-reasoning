"""Write the dataset manifest: every stage's config and outputs with SHA-256, the example
IDs of each split with their application, office action and family, the preference pair
counts and the audit result.

Writes the config's out path (in data/) and a committed copy at its report path. Fails if
an example's family spans splits or a split file doesn't match the splits manifest.

Usage: uv run python scripts/build_manifest.py [configs/dataset.yaml]
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import mlflow

from patent_reasoning.config import load_config
from patent_reasoning.data.splits import SPLITS
from patent_reasoning.schema import SCHEMA_VERSION, Example
from patent_reasoning.tracking import git_commit, start_run


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(path: str) -> dict:
    p = Path(path)
    out = {"path": path, "sha256": sha256(p), "bytes": p.stat().st_size}
    if p.suffix == ".jsonl":
        out["lines"] = sum(1 for _ in open(p))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/dataset.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    p = cfg.params

    splits_dir = Path(p["splits_dir"])
    split_manifest = json.load(open(splits_dir / "manifest.json"))
    family = json.load(open(splits_dir / "families.json"))["application_family"]

    splits, family_splits = {}, {}
    for split in SPLITS:
        path = splits_dir / f"{split}.jsonl"
        assert sha256(path) == split_manifest["splits"][split]["sha256"], f"{split} changed"
        examples = [Example.model_validate_json(line) for line in open(path)]
        rows = []
        for e in examples:
            fam = family[e.provenance.application_id]
            family_splits.setdefault(fam, set()).add(split)
            rows.append([e.example_id, e.provenance.application_id, fam])
        splits[split] = {
            "examples": len(examples),
            "applications": len({r[1] for r in rows}),
            "families": len({r[2] for r in rows}),
            "decision_class_counts": dict(Counter(c for e in examples for c in e.target.decision)),
            "columns": ["example_id", "application_id", "family_id"],
            "rows": sorted(rows),
        }
    assert all(len(s) == 1 for s in family_splits.values()), "family spans splits"

    ids = {s: {row[0] for row in x["rows"]} for s, x in splits.items()}
    pairs = {}
    for path in sorted(Path(p["preferences_dir"]).glob("*.jsonl")):
        rows = [json.loads(line) for line in open(path)]
        assert all(r["example_id"] in ids[r["split"]] for r in rows), f"{path} not in splits"
        pairs[path.stem] = {"pairs": len(rows), "types": dict(Counter(r["type"] for r in rows))}

    audit = json.load(open(p["audit_summary"]))
    manifest = {
        "dataset_version": p["version"],
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit(),
        "stages": {
            name: {
                "config": {"path": s["config"], "sha256": sha256(Path(s["config"]))},
                "outputs": [describe(o) for o in s["outputs"]],
            }
            for name, s in p["stages"].items()
        },
        "splits": splits,
        "preference_pairs": pairs,
        "audit": {k: audit[k] for k in ("sampled", "examples_with_an_error")}
        | {"manual": audit["manual"]["all"]},
    }
    text = json.dumps(manifest, indent=1) + "\n"
    with start_run(cfg):
        Path(p["out"]).write_text(text)
        Path(p["report"]).write_text(text)
        mlflow.log_metrics({f"{s}_examples": v["examples"] for s, v in splits.items()})
        mlflow.log_dict(manifest, "manifest.json")
    print(json.dumps({s: {k: v for k, v in x.items() if k not in ("rows", "columns")}
                      for s, x in splits.items()} | {"pairs": pairs}, indent=2))  # fmt: skip


if __name__ == "__main__":
    main()
