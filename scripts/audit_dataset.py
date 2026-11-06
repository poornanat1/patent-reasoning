"""Audit the split dataset.

Two stages:

- ``sheet``: runs the automatic checks on every example, draws the stratified review
  sample and writes it as Markdown batches (with the office action sentences that name
  each claim) to the config's sheet_dir.
- ``summarize``: validates the manual verdicts (one JSON object per line, fields in
  ``RUBRIC``) and writes the automatic and manual results to the config's report path and
  MLflow. A verdict may carry an ``adjudication`` (fields overridden after a second look,
  with a reason); tallies use the adjudicated values. Each error is also traced into the
  current build: dropped, target changed, or unchanged.

Usage: uv run python scripts/audit_dataset.py {sheet,summarize} [configs/dataset_audit.yaml]
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import mlflow

from patent_reasoning.config import load_config
from patent_reasoning.data.dataset_audit import (
    CHECKS,
    RUBRIC,
    check,
    conflicting_duplicates,
    disposition_sentences,
    final,
    is_error,
    random_sample,
    render,
    sample,
    stratum,
    wilson,
)
from patent_reasoning.data.joined import xml_download_url
from patent_reasoning.data.odp import ODPClient
from patent_reasoning.data.office_action import parse_blocks
from patent_reasoning.data.splits import EVAL_SPLITS, SPLITS
from patent_reasoning.schema import Example
from patent_reasoning.tracking import start_run


def load_splits(splits_dir: str) -> dict[str, list[Example]]:
    return {s: [Example.model_validate_json(line) for line in open(Path(splits_dir) / f"{s}.jsonl")]
            for s in SPLITS}  # fmt: skip


def target_digest(example: Example) -> str:
    target = json.dumps(example.target.model_dump(), sort_keys=True)
    return hashlib.sha256(target.encode()).hexdigest()[:16]


def automatic(examples: dict[str, list[Example]]) -> tuple[dict[str, list[str]], dict]:
    every = [e for xs in examples.values() for e in xs]
    duplicates = conflicting_duplicates(every)
    issues = {e.example_id: check(e) + (["conflicting_duplicate"] * (e.example_id in duplicates))
              for e in every}  # fmt: skip
    by_split = {
        s: {"examples": len(xs), **{c: sum(c in issues[e.example_id] for e in xs) for c in CHECKS}}
        for s, xs in examples.items()
    }
    return issues, by_split


def sheet(p: dict) -> None:
    examples = load_splits(p["splits_dir"])
    issues, by_split = automatic(examples)
    split_of = {e.example_id: s for s, xs in examples.items() for e in xs}
    if p.get("sampling") == "random":
        exclude = {x["example_id"] for path in p.get("exclude_samples", [])
                   for x in json.load(open(path))}  # fmt: skip
        chosen = random_sample(examples, p["quotas"], p["salt"], exclude)
    else:
        chosen = sample(examples, p["quotas"], p["salt"])
    client = ODPClient(p["cache_dir"])
    blocks = {}
    out = Path(p["sheet_dir"])
    out.mkdir(parents=True, exist_ok=True)
    n = p["batches"]
    for i in range(n):
        part = chosen[i::n]
        text = [f"# Audit batch {i + 1} of {n} ({len(part)} examples)\n"]
        for e in part:
            prov = e.provenance
            if prov.office_action_id not in blocks:
                docs = client.documents(prov.application_id)
                ifw = prov.office_action_id
                doc = next((d for d in docs if d.get("documentIdentifier") == ifw), None)
                url = xml_download_url(doc)
                xml = client.document_xml(url) if url else None
                blocks[prov.office_action_id] = parse_blocks(xml) if xml else []
            sentences = disposition_sentences(blocks[prov.office_action_id], e.input.claim.number)
            text.append(render(e, split_of[e.example_id], issues[e.example_id], sentences))
        (out / f"batch-{i + 1}.md").write_text("\n".join(text))
    Path(p["sample"]).write_text(json.dumps(
        [{"example_id": e.example_id, "split": split_of[e.example_id], "stratum": stratum(e),
          "decision": sorted(e.target.decision), "target_sha256": target_digest(e)}
         for e in chosen], indent=1) + "\n")  # fmt: skip
    (out / "automatic.json").write_text(json.dumps(by_split, indent=2) + "\n")
    print(json.dumps({"sampled": dict(Counter(split_of[e.example_id] for e in chosen)),
                      "automatic": by_split}, indent=2))  # fmt: skip


def summarize(p: dict) -> None:
    # Verdicts describe the build the sheet was drawn from; the automatic checks describe
    # the current build, so they show whether fixes made after the review held.
    examples = load_splits(p["splits_dir"])
    _, by_split = automatic(examples)
    drawn = json.load(open(p["sample"]))
    split_of = {x["example_id"]: x["split"] for x in drawn}
    stratum_of = {x["example_id"]: x["stratum"] for x in drawn}
    expected = set(split_of)
    verdicts = [json.loads(line) for line in open(p["verdicts"]) if line.strip()]
    seen = {v["example_id"] for v in verdicts}
    assert seen == expected, f"verdicts don't match the sample: {sorted(seen ^ expected)[:5]}"
    assert len(seen) == len(verdicts), "duplicate verdicts"
    for v in verdicts:
        for field, allowed in RUBRIC.items():
            assert v[field] in allowed, f"{v['example_id']}: {field}={v[field]!r}"
            if field in v.get("adjudication", {}):
                assert v["adjudication"][field] in allowed and v["adjudication"]["reason"]
    adjudicated = [final(v) for v in verdicts]
    current = {e.example_id: e for xs in examples.values() for e in xs}
    digest = {x["example_id"]: x["target_sha256"] for x in drawn}

    def in_current_build(eid: str) -> str:
        if eid not in current:
            return "dropped"
        return "unchanged" if target_digest(current[eid]) == digest[eid] else "target_changed"

    def tally(vs: list[dict]) -> dict:
        return {field: dict(Counter(v[field] for v in vs)) for field in RUBRIC}

    manual = {s: tally([v for v in adjudicated if split_of[v["example_id"]] == s]) for s in SPLITS}
    strata = Counter(stratum_of.values())
    errors = [final(v) for v in verdicts if is_error(v)]

    def rates(vs: list[dict]) -> dict:
        n = len(vs)
        out = {"n": n}
        for name, wrong in (
            ("any_error", is_error),
            ("decision_wrong", lambda v: final(v)["decision"] == "wrong"),
            ("mapping_wrong", lambda v: final(v)["mapping"] == "wrong"),
            ("explanation_wrong", lambda v: final(v)["explanation"] in ("off_claim", "wrong")),
        ):
            k = sum(map(wrong, vs))
            out[name] = {"count": k, "rate": round(k / n, 3) if n else None,
                         "ci95": [round(x, 3) for x in wilson(k, n)]}  # fmt: skip
        return out

    groups = {s: [v for v in verdicts if split_of[v["example_id"]] == s] for s in SPLITS}
    groups["evaluation"] = [v for v in verdicts if split_of[v["example_id"]] in EVAL_SPLITS]
    summary = {
        "sampled": len(verdicts),
        "sampled_by_decision": dict(strata),
        "examples_with_an_error": len(errors),
        "adjudicated": sum("adjudication" in v for v in verdicts),
        "manual_before_adjudication": tally(verdicts),
        "error_rates": ({s: rates(vs) for s, vs in groups.items()}
                        if p.get("sampling") == "random" else None),  # fmt: skip
        "manual": {"all": tally(adjudicated), **manual},
        "errors_in_current_build": dict(Counter(in_current_build(v["example_id"]) for v in errors)),
        "automatic": by_split,
        "errors": [
            {k: v[k] for k in ("example_id", *RUBRIC, "notes")}
            | {"split": split_of[v["example_id"]], "now": in_current_build(v["example_id"])}
            for v in errors
        ],  # fmt: skip
    }
    with start_run(load_config(p["_config"])):
        Path(p["report"]).write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_metrics({"sampled": len(verdicts), "examples_with_an_error": len(errors)})
        mlflow.log_dict(summary, "summary.json")
    print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["sheet", "summarize"])
    parser.add_argument("config", nargs="?", default="configs/dataset_audit.yaml")
    args = parser.parse_args()
    p = {**load_config(args.config).params, "_config": args.config}
    sheet(p) if args.stage == "sheet" else summarize(p)


if __name__ == "__main__":
    main()
