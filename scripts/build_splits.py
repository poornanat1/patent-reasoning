"""Split the SFT corpus into family-disjoint train / val / test / OOD test sets.

Writes to the config's out_dir:

- {train,val,test,ood_test}.jsonl: the examples of each split
- families.json: application -> family, and family -> split
- manifest.json: example IDs and SHA-256 of each split file, plus the parameters

Leakage checks and counts go to the config's report path and MLflow. Train examples whose
prior art includes an evaluation application's own publication are dropped, since that
passage text would repeat the evaluation claim. Evaluation examples the manual audit found
wrong (``audit_verdicts``) are dropped too, so no score is computed against a known-bad
label; audited errors in train stay, as measured noise.

Usage: uv run python scripts/build_splits.py [configs/splits.yaml]
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import mlflow

from patent_reasoning.config import load_config
from patent_reasoning.data.dataset_audit import is_error
from patent_reasoning.data.evidence import normalize_document_id
from patent_reasoning.data.families import families, family_links
from patent_reasoning.data.odp import ODPClient
from patent_reasoning.data.splits import EVAL_SPLITS, SPLITS, assign
from patent_reasoning.schema import Example
from patent_reasoning.tracking import start_run


def own_documents(client: ODPClient, app_id: str) -> set[str]:
    """The application's own publication and patent numbers, normalized."""
    meta = (client.application(app_id) or {}).get("applicationMetaData") or {}
    out = set()
    for raw in (meta.get("earliestPublicationNumber"), meta.get("patentNumber")):
        if norm := normalize_document_id(raw):
            out.add(norm[0])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/splits.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    p = cfg.params
    out_dir = Path(p["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = [Example.model_validate_json(line) for line in open(p["examples"])]
    client = ODPClient(p["cache_dir"])
    apps = sorted({e.provenance.application_id for e in examples})

    with start_run(cfg):
        links = {a: family_links(client, a) for a in apps}
        family_of = families(links)
        split_of_family, split_of_example, dropped = assign(
            examples, family_of, p["salt"], p["val_fraction"], p["test_fraction"],
            p["ood_technology_centers"],
        )  # fmt: skip

        # An evaluation application's own publication cited as prior art in a train input.
        by_id = {e.example_id: e for e in examples}
        eval_docs = {d for e in examples if split_of_example.get(e.example_id) in EVAL_SPLITS
                     for d in own_documents(client, e.provenance.application_id)}  # fmt: skip
        leaking = [eid for eid, s in split_of_example.items() if s == "train"
                   and by_id[eid].input.document_ids() & eval_docs]  # fmt: skip
        for eid in leaking:
            del split_of_example[eid]
        audited_wrong = []
        if p.get("audit_verdicts"):
            wrong = {v["example_id"] for v in map(json.loads, open(p["audit_verdicts"]))
                     if is_error(v)}  # fmt: skip
            audited_wrong = sorted(eid for eid in wrong if split_of_example.get(eid) in EVAL_SPLITS)
            for eid in audited_wrong:
                del split_of_example[eid]

        manifest = {"params": p, "splits": {}}
        for split in SPLITS:
            ids = sorted(eid for eid, s in split_of_example.items() if s == split)
            text = "".join(by_id[eid].model_dump_json() + "\n" for eid in ids)
            (out_dir / f"{split}.jsonl").write_text(text)
            manifest["splits"][split] = {
                "examples": len(ids),
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "example_ids": ids,
            }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (out_dir / "families.json").write_text(json.dumps(
            {"application_family": family_of, "family_split": split_of_family}, indent=2
        ) + "\n")  # fmt: skip

        # Checks: no family or application may span splits.
        split_sets: dict[str, set[str]] = {}
        for eid, split in split_of_example.items():
            e = by_id[eid]
            split_sets.setdefault(family_of[e.provenance.application_id], set()).add(split)
        app_splits: dict[str, set[str]] = {}
        for eid, split in split_of_example.items():
            app_splits.setdefault(by_id[eid].provenance.application_id, set()).add(split)
        train_docs = {d for eid, s in split_of_example.items() if s == "train"
                      for d in by_id[eid].input.document_ids()}  # fmt: skip

        def describe(split: str) -> dict:
            xs = [by_id[eid] for eid, s in split_of_example.items() if s == split]
            return {
                "examples": len(xs),
                "applications": len({e.provenance.application_id for e in xs}),
                "families": len({family_of[e.provenance.application_id] for e in xs}),
                "decision_class_counts": dict(Counter(c for e in xs for c in e.target.decision)),
                "no_rejection_share": round(
                    sum(e.target.decision == {"no_rejection"} for e in xs) / len(xs), 3
                )
                if xs
                else 0,  # fmt: skip
                "evidence_status": dict(Counter(e.target.evidence_status for e in xs)),
                "sharing_prior_art_with_train": sum(
                    bool(e.input.document_ids() & train_docs) for e in xs
                )
                if split != "train"
                else None,  # fmt: skip
            }

        sizes = Counter(family_of.values())
        summary = {
            "examples_in": len(examples),
            "applications": len(apps),
            "families": len(sizes),
            "families_with_several_applications": sum(n > 1 for n in sizes.values()),
            "applications_in_shared_families": sum(n for n in sizes.values() if n > 1),
            "applications_with_family_links": sum(len(v) > 1 for v in links.values()),
            "dropped_from_eval_not_fully_grounded": len(dropped),
            "dropped_from_train_cites_eval_application": len(leaking),
            "dropped_from_eval_audited_wrong": audited_wrong,
            "families_spanning_splits": sum(len(v) > 1 for v in split_sets.values()),
            "applications_spanning_splits": sum(len(v) > 1 for v in app_splits.values()),
            "splits": {s: describe(s) for s in SPLITS},
        }  # fmt: skip
        assert summary["families_spanning_splits"] == 0, "family leak"
        assert summary["applications_spanning_splits"] == 0, "application leak"
        Path(p["report"]).write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, int | float)})
        mlflow.log_metrics({f"{s}_examples": v["examples"] for s, v in summary["splits"].items()})
        mlflow.log_dict(summary, "summary.json")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
