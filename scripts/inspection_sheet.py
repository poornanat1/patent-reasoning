"""Write a compact Markdown sheet of joined records for manual review.

For each record: the rejected claim text, the rejection statement, and each resolved pin
cite shown next to the examiner's sentence that uses it and the cited paragraph's text.

Usage: uv run python scripts/inspection_sheet.py [--n 50] [--records path] [--out path]
"""

import argparse
import hashlib
import json
from pathlib import Path


def clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def context(reasoning: str, evidence: dict, width: int = 260) -> str:
    i = evidence.get("offset", -1)
    if i < 0:
        i = reasoning.find(evidence["raw"])
    if i < 0:
        return ""
    end = i + len(evidence["raw"])
    return clip(reasoning[max(0, i - width) : end], width + len(evidence["raw"]) + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="data/interim/joined_sample/records.jsonl")
    parser.add_argument("--out", default="data/interim/joined_sample/inspection.md")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--salt", default="inspect-v1")
    args = parser.parse_args()

    records = [json.loads(line) for line in open(args.records)]
    records = [r for r in records if r["status"]["traceable"]]

    def order(r):
        key = f"{args.salt}:{r['office_action_id']}:{r['rejection_type']}"
        return hashlib.sha256(key.encode()).hexdigest()

    records.sort(key=order)

    lines = [f"# Inspection sheet ({min(args.n, len(records))} traceable records)\n"]
    for i, r in enumerate(records[: args.n], 1):
        refs = {x["document_id"]: x for x in r["references"]}
        claim_no = min(r["claims"], key=int)
        flags = [k for k, v in r["status"].items() if v]
        lines += [
            f"## {i}. {r['application_id']} / {r['office_action_id']} / §{r['rejection_type']}",
            f"- mail {r['mail_date']}, claims {r['oard_claims'][:12]}, amended_before="
            f"{r['claims_amended_before_action']}, source={r['claims_source']}",
            f"- status: {', '.join(flags)}",
            f"- refs: {[(d, x['alias'], x['fulltext']) for d, x in refs.items()]}",
            f"- **statement**: {clip(' | '.join(r['statements']), 400)}",
            f"- **claim {claim_no}**: {clip(r['claims'][claim_no], 450)}",
        ]
        resolved = [e for e in r["evidence"] if e["resolved"]][:2]
        for e in resolved:
            alias = (refs.get(e["document_id"]) or {}).get("alias")
            lines += [
                f"- **cite** {e['raw']!r} → {e['document_id']} ({alias})",
                f"  - examiner: …{context(r['examiner_reasoning'], e)}",
                f"  - paragraph: {clip(e['text'], 350)}",
            ]
        if not resolved:
            lines.append(f"- **reasoning**: {clip(r['examiner_reasoning'], 500)}")
        lines.append("")
    Path(args.out).write_text("\n".join(lines))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
