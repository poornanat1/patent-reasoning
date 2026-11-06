"""Turn claim records and their evidence into schema-valid ``Example``s for SFT.

A record becomes an example only when its target is grounded in what the model will see:

- the label is trustworthy: reliable claim text, no conflicting dispositions, and every
  102/103 ground resolved to cited documents
- a 102/103 rejection has at least one of the examiner's cited passages in the input, so
  the target never asserts a disclosure the model can't check, and an explanation that
  restates at least one of the claim's own limitations (so it isn't about another claim)
- a no-rejection example has cited art with text to judge the claim against

Everything else is excluded with a reason, so unresolved labels stay visible instead of
silently disappearing.

Target construction:

- ``grounds``: the record's grounds, deduplicated
- ``claim_mapping``: each of the claim's own limitations that the examiner tied to a pin
  cite, with the cited passages that are in the input. An allowable dependent claim with a
  single limitation maps it as not disclosed, which is what the examiner concluded; other
  no-rejection claims get no mapping, because the examiner doesn't say which limitation
  the art lacks
- ``explanation``: the examiner's reasoning for the claim (or reasons for allowance)
- ``evidence_status``: ``supported`` when every limitation is mapped, otherwise
  ``partially_supported``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from patent_reasoning.data.claim_records import word_coverage
from patent_reasoning.data.evidence import approx_tokens
from patent_reasoning.schema import (
    PRIOR_ART_STATUTES,
    ClaimText,
    Example,
    ExampleInput,
    Ground,
    LimitationMapping,
    ModelOutput,
    Passage,
    PinCite,
    PriorArtDocument,
    Provenance,
)

# Share of a limitation's content words a mapped passage must contain (see ``_mapping``).
PASSAGE_FLOOR = 0.1
# A listing that shows only "4. (canceled)": the action's label has no claim text to apply to.
_CANCELED = re.compile(r"^\s*\d+\s*\.?\s*\(\s*cancell?ed\s*\)", re.IGNORECASE)

# Exclusion reasons, in the order they are checked.
REASONS = (
    "claim_text_unreliable",
    "claim_canceled",
    "conflicting_disposition",
    "non_patent_literature",
    "references_unresolved",
    "ground_reference_not_cited",
    "no_prior_art_text",
    "evidence_not_in_input",
    "explanation_off_claim",
    "target_too_long",
    "schema_invalid",
)


@dataclass
class Outcome:
    example_id: str
    example: Example | None = None
    reason: str | None = None  # why it was excluded
    detail: str = ""
    flags: list[str] = field(default_factory=list)


def _input(record: dict, documents: dict[str, dict], passage_ids: list[str]) -> ExampleInput:
    chosen = set(passage_ids)
    prior_art = []
    for doc_id in record["cited_documents"]:
        doc = documents[doc_id]
        prior_art.append(
            PriorArtDocument(
                document_id=doc_id,
                kind=doc["kind"],
                title=doc["title"],
                passages=[
                    Passage(passage_id=p["passage_id"], locator=p["locator"], text=p["text"])
                    for p in doc["passages"]
                    if p["passage_id"] in chosen
                ],
            )
        )
    return ExampleInput(
        claim=ClaimText(number=record["claim_number"], text=record["claim_text"]),
        parent_claims=[
            ClaimText(number=n, text=t)
            for n, t in zip(record["parent_claims"], record["parent_claim_texts"], strict=True)
        ],
        prior_art=prior_art,
    )


def _grounds(record: dict) -> list[Ground]:
    seen, out = set(), []
    for g in record["grounds"]:
        key = (g["statute"], tuple(g["references"]))
        if key not in seen:
            seen.add(key)
            out.append(Ground(statute=g["statute"], references=g["references"]))
    return out


def _mapping(record: dict, visible: dict[str, str], docs: set[str]) -> list[LimitationMapping]:
    """Limitations the examiner tied to a pin cite whose passages are in the input.

    ``visible`` is passage ID -> text. A cited passage that shares fewer than
    ``PASSAGE_FLOOR`` of the limitation's content words is left out: the audit found these
    are almost always a cite resolved to the wrong document, or the part of a cited range
    about something else.
    """
    out = []
    for i, limitation in enumerate(record["limitations"]):
        by_doc: dict[str, list[str]] = {}
        for e in record["evidence"]:
            if i not in e["limitations"] or e["document_id"] not in docs:
                continue
            ids = [f"{e['document_id']}:{p}" for p in e["paragraphs"]]
            by_doc.setdefault(e["document_id"], []).extend(
                x
                for x in ids
                if x in visible and word_coverage(limitation, visible[x]) >= PASSAGE_FLOOR
            )
        best = max(by_doc.items(), key=lambda kv: len(kv[1]), default=None)
        if best and best[1]:
            doc_id, passage_ids = best
            out.append(LimitationMapping(limitation=limitation, disclosed=True,
                                         document_id=doc_id,
                                         passage_ids=list(dict.fromkeys(passage_ids))))  # fmt: skip
    return out


def _pin_cites(record: dict, visible: set[str]) -> list[PinCite]:
    return [
        PinCite(
            raw=e["raw"],
            kind=e["kind"],
            document_id=e["document_id"],
            resolved_passage_ids=[
                pid for p in e["paragraphs"] if (pid := f"{e['document_id']}:{p}") in visible
            ],
        )
        for e in record["evidence"]
    ]


def build_example(
    record: dict,
    documents: dict[str, dict],
    passage_ids: list[str],
    condition: str,
    max_target_tokens: int,
) -> Outcome:
    out = Outcome(record["example_id"], flags=list(record["flags"]))
    flags = set(record["flags"])
    if not record["claim_text_reliable"]:
        out.reason = "claim_text_unreliable"
    elif _CANCELED.match(record["claim_text"]):
        out.reason = "claim_canceled"
    elif "conflicting_disposition" in flags:
        out.reason = "conflicting_disposition"
    elif "ground_non_patent_literature" in flags:
        out.reason = "non_patent_literature"
    elif "ground_references_unresolved" in flags:
        out.reason = "references_unresolved"
    elif "ground_reference_not_cited" in flags:
        out.reason = "ground_reference_not_cited"
    if out.reason:
        return out

    inp = _input(record, documents, passage_ids)
    visible = inp.passage_ids()
    docs = inp.document_ids()
    grounds = _grounds(record)
    has_art = any(g.statute in PRIOR_ART_STATUTES for g in grounds)
    if (has_art or not grounds) and not visible:
        out.reason = "no_prior_art_text"
        return out

    examiner_cited = {
        f"{e['document_id']}:{p}" for e in record["evidence"] for p in e["paragraphs"]
    }
    mapping: list[LimitationMapping] = []
    status = "not_applicable"
    if has_art:
        if not examiner_cited & visible:
            out.reason = "evidence_not_in_input"
            out.detail = "column/line cites only" if not examiner_cited else "cited passages cut"
            return out
        text = record["examiner_text"]
        if (
            text
            and record["limitations"]
            and max(word_coverage(lim, text) for lim in record["limitations"]) < 0.5
        ):
            out.reason = "explanation_off_claim"
            return out
        texts = {p.passage_id: p.text for d in inp.prior_art for p in d.passages}
        mapping = _mapping(record, texts, docs)
        complete = bool(record["limitations"]) and len(mapping) == len(record["limitations"])
        status = "supported" if complete else "partially_supported"
        if not complete:
            out.flags.append("mapping_partial")
    elif not grounds:
        if record["decision_source"] == "allowable_dependent" and len(record["limitations"]) == 1:
            mapping = [LimitationMapping(limitation=record["limitations"][0], disclosed=False)]
        else:
            out.flags.append("mapping_unknown")

    explanation = record["examiner_text"] or None
    try:
        target = ModelOutput(grounds=grounds, claim_mapping=mapping, explanation=explanation,
                             evidence_status=status)  # fmt: skip
        provenance = Provenance(
            application_id=record["application_id"],
            office_action_id=record["office_action_id"],
            document_code=record["document_code"],
            mail_date=record["mail_date"],
            art_unit=record["art_unit"],
            uspc_class=record["uspc_class"],
            decision_source=record["decision_source"],
            oard_agrees=record["oard_agrees"],
            claims_source=record["claims_source"],
            claims_amended_before_action=record["claims_amended_before_action"],
            claim_text_verified=record["claim_text_verified"],
            evidence_condition=condition,
            examiner_text=record["examiner_text"],
            pin_cites=_pin_cites(record, visible),
            explanation_source=record["explanation_source"],
            flags=sorted(set(out.flags)),
        )
        example = Example(example_id=record["example_id"], input=inp, target=target,
                          provenance=provenance)  # fmt: skip
    except ValidationError as e:
        out.reason, out.detail = "schema_invalid", str(e).splitlines()[0]
        return out
    if approx_tokens(target.model_dump_json()) > max_target_tokens:
        out.reason = "target_too_long"
        return out
    out.example = example
    return out


def target_tokens(example: Example) -> int:
    return approx_tokens(example.target.model_dump_json())


def summarize_outcomes(outcomes: list[Outcome], records: dict[str, dict[str, Any]]) -> dict:
    from collections import Counter

    kept = [o for o in outcomes if o.example]
    tokens = sorted(target_tokens(o.example) for o in kept)

    def decision(o: Outcome) -> str:
        return "+".join(sorted(o.example.target.decision))

    return {
        "records": len(outcomes),
        "examples": len(kept),
        "excluded": len(outcomes) - len(kept),
        "excluded_by_reason": {r: sum(o.reason == r for o in outcomes) for r in REASONS},
        "evidence_not_in_input_detail": dict(
            Counter(o.detail for o in outcomes if o.reason == "evidence_not_in_input")
        ),
        "schema_invalid_examples": [o.detail for o in outcomes if o.reason == "schema_invalid"][:5],
        "decisions": dict(Counter(decision(o) for o in kept).most_common()),
        "decision_class_counts": dict(Counter(c for o in kept for c in o.example.target.decision)),
        "decision_source": dict(Counter(o.example.provenance.decision_source for o in kept)),
        "evidence_status": dict(Counter(o.example.target.evidence_status for o in kept)),
        "with_explanation": sum(o.example.target.explanation is not None for o in kept),
        "explanation_source": dict(Counter(o.example.provenance.explanation_source for o in kept)),
        "with_claim_mapping": sum(bool(o.example.target.claim_mapping) for o in kept),
        "mapped_limitations": sum(len(o.example.target.claim_mapping) for o in kept),
        "oard_disagrees": sum(o.example.provenance.oard_agrees is False for o in kept),
        "applications": len({o.example.provenance.application_id for o in kept}),
        "target_tokens_p50_p95_max": [
            tokens[len(tokens) // 2],
            tokens[int(len(tokens) * 0.95)],
            tokens[-1],
        ]
        if tokens
        else [],  # fmt: skip
        "flags": dict(Counter(f for o in kept for f in o.example.provenance.flags).most_common()),
        "excluded_records_by_decision_source": dict(
            Counter(records[o.example_id]["decision_source"] for o in outcomes if not o.example)
        ),
    }
