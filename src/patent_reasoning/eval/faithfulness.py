"""Citation checks: does what the model cites exist, and does it support the limitation?

Per example, from the output's claim mapping and grounds:

- **Existence.** Every cited document must be in the input, and every cited passage must
  be in the input and belong to the document it is cited for. Generation doesn't constrain
  IDs, so a model can cite something it never saw.
- **Lexical support.** A cited passage that exists supports its limitation if it shares at
  least ``SUPPORT_FLOOR`` of the limitation's content words, the floor the reference
  mappings use. This is a lexical proxy: it catches citations about something else, not
  passages that use the same words to say something different.
- **Unsupported assertions.** A limitation marked disclosed with no valid passage, and a
  102/103 rejection with no valid passage at all.
- **Overclaimed support.** Evidence status "supported" when some limitation isn't marked
  disclosed with a valid passage.
- **Agreement with the examiner.** For limitations the reference mapping ties to passages
  (the examiner's resolved pin cites), whether the model marks them disclosed and cites at
  least one of the same passages.

Outputs that aren't JSON contribute nothing here; they're counted by the decision metrics.
"""

from __future__ import annotations

import json
from collections import Counter

from patent_reasoning.data.claim_records import word_coverage
from patent_reasoning.schema import PRIOR_ART_STATUTES, Example

SUPPORT_FLOOR = 0.1


def _data(text: str) -> dict | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def check(example: Example, text: str) -> Counter:
    """Counts for one example; ``score`` turns summed counts into rates."""
    out: Counter = Counter()
    data = _data(text)
    if data is None:
        return out
    out["readable"] += 1
    docs = example.input.document_ids()
    passages = {p.passage_id: p.text for d in example.input.prior_art for p in d.passages}
    mapping = [m for m in data.get("claim_mapping") or [] if isinstance(m, dict)]
    grounds = [g for g in data.get("grounds") or [] if isinstance(g, dict)]

    for g in grounds:
        for ref in g.get("references") or []:
            out["references_cited"] += 1
            out["references_missing"] += ref not in docs
    valid_any = False
    fully_mapped = bool(mapping)
    for m in mapping:
        doc, pids = m.get("document_id"), m.get("passage_ids") or []
        if doc is not None:
            out["documents_cited"] += 1
            out["documents_missing"] += doc not in docs
        valid = []
        for pid in pids:
            out["passages_cited"] += 1
            if pid not in passages:
                out["passages_missing"] += 1
            elif doc is None or not pid.startswith(f"{doc}:"):
                out["passages_wrong_document"] += 1
            else:
                valid.append(pid)
                supported = word_coverage(m.get("limitation", ""), passages[pid]) >= SUPPORT_FLOOR
                out["passages_supporting"] += supported
        if m.get("disclosed"):
            out["limitations_disclosed"] += 1
            out["disclosed_without_valid_passage"] += not valid
        valid_any |= bool(valid)
        fully_mapped &= bool(m.get("disclosed")) and bool(valid)

    if any(g.get("statute") in PRIOR_ART_STATUTES for g in grounds):
        out["art_rejections"] += 1
        out["art_rejections_without_valid_passage"] += not valid_any
        if data.get("evidence_status") == "supported":
            out["supported_claimed"] += 1
            out["supported_overclaimed"] += not fully_mapped

    predicted = {m.get("limitation"): m for m in mapping}
    for ref in example.target.claim_mapping:
        if not (ref.disclosed and ref.passage_ids):
            continue
        out["examiner_pinned_limitations"] += 1
        m = predicted.get(ref.limitation)
        if m and m.get("disclosed"):
            out["examiner_pinned_disclosed"] += 1
            cited = set(m.get("passage_ids") or [])
            out["examiner_pinned_hit"] += bool(cited & set(ref.passage_ids))
            out["examiner_pinned_model_passages"] += len(cited)
            out["examiner_pinned_model_passages_in_examiner"] += len(cited & set(ref.passage_ids))
    return out


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def score(counts: Counter) -> dict:
    c = counts
    return {
        "citation_existence": _rate(
            c["passages_cited"] - c["passages_missing"] - c["passages_wrong_document"],
            c["passages_cited"],
        ),
        "invalid_passage_rate": _rate(
            c["passages_missing"] + c["passages_wrong_document"], c["passages_cited"]
        ),
        "invalid_reference_rate": _rate(c["references_missing"], c["references_cited"]),
        # Cited passages that exist and lexically support their limitation.
        "citation_precision": _rate(c["passages_supporting"], c["passages_cited"]),
        "unsupported_assertion_rate": _rate(
            c["disclosed_without_valid_passage"], c["limitations_disclosed"]
        ),
        "art_rejection_without_citation_rate": _rate(
            c["art_rejections_without_valid_passage"], c["art_rejections"]
        ),
        "support_overclaim_rate": _rate(c["supported_overclaimed"], c["supported_claimed"]),
        "examiner_pin_recall": _rate(c["examiner_pinned_hit"], c["examiner_pinned_limitations"]),
        "examiner_pin_precision": _rate(
            c["examiner_pinned_model_passages_in_examiner"], c["examiner_pinned_model_passages"]
        ),
        "counts": dict(c),
    }
