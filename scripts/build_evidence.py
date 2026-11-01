"""Build the evidence dataset: cited documents as passages, and each claim's retrieved set.

Reads the claim records, then writes to the config's out_dir:

- documents.jsonl: every cited document with its normalized ID, kind and passages
- retrieval.jsonl: per claim record, the passages selected under the fixed budget in each
  evidence condition (see patent_reasoning.data.evidence), plus the examiner's resolved
  pin-cite passages for the claim and which of them made it into the input

The summary (with recall at several budgets and prompt lengths) goes to the config's
report path and MLflow.

Usage: uv run python scripts/build_evidence.py [configs/evidence.yaml]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import mlflow

from patent_reasoning.config import load_config
from patent_reasoning.data.evidence import (
    Budget,
    PassageRecord,
    approx_tokens,
    claim_query,
    document_kind,
    document_passages,
    select_passages,
    terms,
)
from patent_reasoning.data.joined import Joiner
from patent_reasoning.data.odp import ODPClient
from patent_reasoning.tracking import start_run


def percentile(values: list[int], q: float) -> int:
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))] if values else 0


def gold_passages(record: dict) -> set[str]:
    return {f"{e['document_id']}:{p}" for e in record["evidence"] for p in e["paragraphs"]}


def recall_stats(rows: list[tuple[set[str], list[str]]]) -> dict:
    """Pin-cite recall over records that have resolved examiner pin cites."""
    rows = [(g, set(s)) for g, s in rows if g]
    if not rows:
        return {"records": 0}
    return {
        "records": len(rows),
        "passage_recall": round(sum(len(g & s) for g, s in rows) / sum(len(g) for g, _ in rows), 3),
        "any_hit_rate": round(sum(bool(g & s) for g, s in rows) / len(rows), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/evidence.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    p = cfg.params
    out_dir = Path(p["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = Budget(**p["budget"])

    records = [json.loads(line) for line in open(p["claim_records"])]
    joiner = Joiner(ODPClient(p["cache_dir"]), None)

    with start_run(cfg):
        documents: dict[str, dict] = {}
        for doc_id in sorted({d for r in records for d in r["cited_documents"]}):
            kind = document_kind(doc_id)
            source_app, pub = (None, None)
            if doc_id.isdigit():
                source_app, pub = joiner.fetch_reference(doc_id)
            documents[doc_id] = {
                "document_id": doc_id,
                "kind": kind,
                "title": pub.title if pub else "",
                "source_application": source_app,
                "fulltext": pub is not None,
                "passages": [vars(x) for x in document_passages(doc_id, pub,
                                                                budget.max_passage_tokens)],
            }  # fmt: skip
        with open(out_dir / "documents.jsonl", "w") as f:
            f.writelines(json.dumps(d) + "\n" for d in documents.values())

        passages = {d: [PassageRecord(**x) for x in doc["passages"]]
                    for d, doc in documents.items()}  # fmt: skip
        cited_pool = {f["office_action_id"]: frozenset(f.get("cited_passages", []))
                      for f in map(json.loads, open(p["claim_funnel"]))}  # fmt: skip
        conditions = ("action_cited", "retrieved")
        rows = {c: [] for c in conditions}
        compare = {c: {n: [] for n in p["compare_max_tokens"]} for c in conditions}
        retrieval, prompt_tokens = [], []
        for r in records:
            docs = {d: passages[d] for d in r["cited_documents"] if passages[d]}
            query = claim_query(r["claim_text"], r["parent_claim_texts"], budget.claim_weight)
            lim_queries = [terms(lim) for lim in r["limitations"]]
            pool = cited_pool.get(r["office_action_id"], frozenset())
            gold = gold_passages(r)
            selected = {}
            for c in conditions:
                cited = pool if c == "action_cited" else frozenset()
                selected[c] = select_passages(query, docs, budget, lim_queries, cited)
                rows[c].append((gold, selected[c].passage_ids))
                for n in compare[c]:
                    alt = Budget(**{**p["budget"], "max_tokens": n, "max_passages": 10_000})
                    sel = select_passages(query, docs, alt, lim_queries, cited)
                    compare[c][n].append((gold, sel.passage_ids))
            primary = selected[p["condition"]]
            claim_tokens = approx_tokens(r["claim_text"]) + sum(
                approx_tokens(t) for t in r["parent_claim_texts"]
            )
            prompt_tokens.append(claim_tokens + primary.tokens + p["prompt_reserve_tokens"])
            retrieval.append({
                "example_id": r["example_id"],
                "condition": p["condition"],
                "passage_ids": primary.passage_ids,
                "evidence_tokens": primary.tokens,
                **{f"{c}_passage_ids": selected[c].passage_ids for c in conditions},
                "claim_tokens": claim_tokens,
                "documents_with_text": len(docs),
                "documents_cited": len(r["cited_documents"]),
                "action_cited_pool_size": len(pool),
                "gold_passage_ids": sorted(gold),
                "gold_in_input": sorted(gold & set(primary.passage_ids)),
            })  # fmt: skip
        with open(out_dir / "retrieval.jsonl", "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in retrieval)

        docs_list = list(documents.values())
        claim_tokens = [x["claim_tokens"] for x in retrieval]
        no_text = [x for x in retrieval if x["documents_with_text"] == 0]
        by_quality = Counter(r["quality"] for r in records)
        summary = {
            "budget": vars(budget),
            "documents": len(docs_list),
            "documents_by_kind": dict(Counter(d["kind"] for d in docs_list)),
            "documents_with_fulltext": dict(Counter(d["kind"] for d in docs_list if d["fulltext"])),
            "passages": sum(len(d["passages"]) for d in docs_list),
            "passages_per_document_median": percentile(
                [len(d["passages"]) for d in docs_list if d["passages"]], 0.5
            ),
            "records": len(records),
            "records_without_any_passages": len(no_text),
            "records_without_any_passages_by_quality": dict(
                Counter(r["quality"] for r, x in zip(records, retrieval, strict=True)
                        if x["documents_with_text"] == 0)  # fmt: skip
            ),
            "records_by_quality": dict(by_quality),
            "evidence_tokens_median": percentile([x["evidence_tokens"] for x in retrieval], 0.5),
            "claim_tokens_p50_p95_max": [percentile(claim_tokens, 0.5),
                                         percentile(claim_tokens, 0.95), max(claim_tokens)],
            "prompt_reserve_tokens": p["prompt_reserve_tokens"],
            "max_seq_len": p["max_seq_len"],
            "records_within_max_seq_len": sum(t <= p["max_seq_len"] for t in prompt_tokens),
            "condition": p["condition"],
            "action_cited_pool_size_median": percentile(
                [x["action_cited_pool_size"] for x in retrieval], 0.5
            ),
            "pin_cite_recall": {c: recall_stats(rows[c]) for c in conditions},
            "pin_cite_recall_by_max_tokens": {
                c: {str(n): recall_stats(v) for n, v in compare[c].items()} for c in conditions
            },
        }  # fmt: skip
        Path(p["report"]).write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, int | float)})
        mlflow.log_metrics({f"recall_{c}_{k}": v for c, stats in summary["pin_cite_recall"].items()
                            for k, v in stats.items()})  # fmt: skip
        mlflow.log_dict(summary, "summary.json")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
