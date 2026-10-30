"""Build the joined feasibility set from the OARD sample and the Open Data Portal.

Writes records.jsonl (one per office action x 102/103 rejection), funnel.jsonl (one per
office action) and summary.json to the config's out_dir, and logs the summary to MLflow.

Usage: uv run python scripts/build_joined_sample.py [configs/joined_sample.yaml] [--limit N]
"""

import argparse
import json
from pathlib import Path

import mlflow

from patent_reasoning.config import load_config
from patent_reasoning.data.joined import Joiner, select_office_actions
from patent_reasoning.data.oard import TABLES, connect
from patent_reasoning.data.odp import ODPClient
from patent_reasoning.tracking import start_run

RECORD_FLAGS = ("statement_found", "claims_found", "claims_reliable", "prior_art_text",
                "paragraph_evidence", "traceable", "evidence_grounded")  # fmt: skip
FUNNEL_FLAGS = ("oa_in_file_wrapper", "oa_xml_available", "oa_text", "published_claims",
                "claims_amended_before_action")  # fmt: skip


def summarize(records: list[dict], funnel: list[dict], client: ODPClient) -> dict:
    refs = [r for rec in records for r in rec["references"]]
    us_refs = [r for r in refs if r["kind"] != "foreign"]
    evidence = [e for rec in records for e in rec["evidence"]]
    paragraph = [e for e in evidence if e["kind"] == "paragraph"]
    return {
        "office_actions": len(funnel),
        **{f"oa_{k}": sum(bool(f.get(k)) for f in funnel) for k in FUNNEL_FLAGS},
        "records": len(records),
        "records_by_type": {
            t: sum(r["rejection_type"] == t for r in records) for t in ("102", "103")
        },
        **{f"records_{k}": sum(r["status"][k] for r in records) for k in RECORD_FLAGS},
        "records_claims_as_filed_ocr": sum(r["claims_source"] == "as_filed_ocr" for r in records),
        "references": len(refs),
        "references_foreign": len(refs) - len(us_refs),
        "references_us_fulltext": sum(r["fulltext"] for r in us_refs),
        "references_us": len(us_refs),
        "references_in_oard_and_text": sum(r["in_oard"] and r["in_text"] for r in refs),
        "references_oard_only": sum(r["in_oard"] and not r["in_text"] for r in refs),
        "references_text_only": sum(r["in_text"] and not r["in_oard"] for r in refs),
        "pin_cites": len(evidence),
        "pin_cites_column_line": sum(e["kind"] == "column_line" for e in evidence),
        "pin_cites_paragraph": len(paragraph),
        "pin_cites_attributed": sum(e["document_id"] is not None for e in evidence),
        "pin_cites_paragraph_resolved": sum(e["resolved"] for e in paragraph),
        "api_requests": client.requests_made,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/joined_sample.yaml")
    parser.add_argument("--limit", type=int, help="override n_office_actions (smoke tests)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    p = cfg.params
    out_dir = Path(p["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    con = connect({t: Path(p["sample_dir"]) / f"{t}.parquet" for t in TABLES})
    client = ODPClient(p["cache_dir"])
    joiner = Joiner(client, con)
    n = args.limit or p["n_office_actions"]
    actions = select_office_actions(con, n, p["year_min"], p["year_max"], p["salt"])

    with start_run(cfg):
        records, funnel = [], []
        for i, action in enumerate(actions, 1):
            try:
                recs, f = joiner.join(action)
            except Exception as e:  # keep going; failures are part of the funnel
                recs, f = [], {"office_action_id": action["ifw_number"], "error": repr(e)}
            records += [r.to_dict() for r in recs]
            funnel.append(f)
            if i % 10 == 0 or i == len(actions):
                done = sum(r["status"]["traceable"] for r in records)
                print(f"{i}/{len(actions)} actions, {len(records)} records, {done} traceable",
                      flush=True)  # fmt: skip

        with open(out_dir / "records.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
        with open(out_dir / "funnel.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in funnel)
        summary = summarize(records, funnel, client)
        summary["errors"] = sum("error" in f for f in funnel)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, int)})
        mlflow.log_dict(summary, "summary.json")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
