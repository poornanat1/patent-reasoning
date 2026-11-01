"""Build claim-level records from the OARD sample, office action text and published claims.

Writes claims.jsonl (one per claim per office action) and funnel.jsonl (one per office
action) to the config's out_dir, the summary to the config's report path, and logs the
summary to MLflow.

Usage: uv run python scripts/build_claim_records.py [configs/claim_records.yaml] [--limit N]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import mlflow

from patent_reasoning.config import load_config
from patent_reasoning.data.claim_records import RESTATEMENT_THRESHOLD, ClaimRecordBuilder
from patent_reasoning.data.joined import Joiner, select_office_actions
from patent_reasoning.data.oard import TABLES, connect
from patent_reasoning.data.odp import ODPClient
from patent_reasoning.tracking import start_run


def _share(n: int, d: int) -> float:
    return round(n / d, 3) if d else 0.0


def summarize(records: list[dict], funnel: list[dict]) -> dict:
    art = [r for r in records if set(r["decision"]) & {"102", "103"}]
    negatives = [r for r in records if r["decision"] == ["no_rejection"]]
    evidence = [e for r in records for e in r["evidence"]]
    covered = [
        len({i for e in r["evidence"] for i in e["limitations"]}) / len(r["limitations"])
        for r in art if r["limitations"]
    ]  # fmt: skip
    overlaps = sorted(r["restatement_overlap"] for r in records
                      if r["restatement_overlap"] is not None)  # fmt: skip
    oard = [r for r in records if r["oard_agrees"] is not None]
    return {
        "office_actions": len(funnel),
        "office_actions_with_text": sum(bool(f.get("oa_text")) for f in funnel),
        "office_actions_with_claims": sum(bool(f.get("claims_source")) for f in funnel),
        "office_actions_with_records": len({r["office_action_id"] for r in records}),
        "errors": sum("error" in f for f in funnel),
        "claims_in_dispositions": sum(f.get("claims_in_dispositions", 0) for f in funnel),
        "claims_missing_text": sum(f.get("claims_missing_text", 0) for f in funnel),
        "records": len(records),
        "applications": len({r["application_id"] for r in records}),
        "decisions": dict(Counter("+".join(r["decision"]) for r in records).most_common()),
        "decision_source": dict(Counter(r["decision_source"] for r in records)),
        "decision_class_counts": dict(Counter(c for r in records for c in r["decision"])),
        "negatives": len(negatives),
        "negatives_share": _share(len(negatives), len(records)),
        "art_allowable_with_112_only": sum(
            r["art_allowable"] and "102" not in r["decision"] and "103" not in r["decision"]
            for r in records
        ),  # fmt: skip
        "oard_covered": len(oard),
        "oard_agrees": sum(r["oard_agrees"] for r in oard),
        "oard_agreement_rate": _share(sum(r["oard_agrees"] for r in oard), len(oard)),
        "grounds_invalid": sum(not g["valid"] for r in records for g in r["grounds"]),
        "grounds": sum(len(r["grounds"]) for r in records),
        "claims_source": dict(Counter(r["claims_source"] for r in records)),
        "claims_amended_before_action": sum(r["claims_amended_before_action"] for r in records),
        "pending_status": dict(Counter(str(r["pending_status"]) for r in records)),
        "claim_text_verified": dict(Counter(str(r["claim_text_verified"]) for r in records)),
        "claim_text_reliable": sum(r["claim_text_reliable"] for r in records),
        "restatement_threshold": RESTATEMENT_THRESHOLD,
        "restatement_checked": len(overlaps),
        "restatement_overlap_quartiles": [
            overlaps[int(len(overlaps) * q)] for q in (0.25, 0.5, 0.75)
        ]
        if overlaps
        else [],
        "explanation_source": dict(Counter(r["explanation_source"] for r in records)),
        "limitations_per_claim": _share(sum(len(r["limitations"]) for r in records), len(records)),
        "art_rejections": len(art),
        "art_rejections_with_paragraph_evidence": sum(
            any(e["paragraphs"] for e in r["evidence"]) for r in art
        ),
        "art_limitation_coverage_mean": _share(sum(covered), len(covered)),
        "pin_cites": len(evidence),
        "pin_cites_paragraph_resolved": sum(bool(e["paragraphs"]) for e in evidence),
        "pin_cites_mapped_to_limitation": sum(bool(e["limitations"]) for e in evidence),
        "quality": dict(Counter(r["quality"] for r in records)),
        "quality_by_decision_source": {
            src: dict(Counter(r["quality"] for r in records if r["decision_source"] == src))
            for src in ("rejection_statement", "allowable_dependent", "allowed")
        },
        "flags": dict(Counter(f for r in records for f in r["flags"]).most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/claim_records.yaml")
    parser.add_argument("--limit", type=int, help="override n_office_actions (smoke tests)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    p = cfg.params
    out_dir = Path(p["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    con = connect({t: Path(p["sample_dir"]) / f"{t}.parquet" for t in TABLES})
    client = ODPClient(p["cache_dir"])
    builder = ClaimRecordBuilder(Joiner(client, con), con)
    n = args.limit or p["n_office_actions"]
    actions = select_office_actions(
        con, n, p["year_min"], p["year_max"], p["salt"], first_only=p["first_only"]
    )

    with start_run(cfg):
        records, funnel = [], []
        for i, action in enumerate(actions, 1):
            try:
                recs, f = builder.build(action)
            except Exception as e:  # keep going; failures are part of the funnel
                recs, f = [], {"office_action_id": action["ifw_number"], "error": repr(e)}
            records += [r.to_dict() for r in recs]
            funnel.append(f)
            if i % 50 == 0 or i == len(actions):
                print(f"{i}/{len(actions)} actions, {len(records)} claim records", flush=True)

        with open(out_dir / "claims.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
        with open(out_dir / "funnel.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in funnel)
        summary = summarize(records, funnel)
        summary["api_requests"] = client.requests_made
        Path(p["report"]).write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, int | float)})
        mlflow.log_dict(summary, "summary.json")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
