"""Preference pairs for DPO, built from SFT examples.

The chosen output is always the example's target (what the examiner did, grounded in the
input). The rejected output is a programmatic corruption of it. Every rejected output is a
valid ``ModelOutput`` that only cites documents and passages present in the input, so a
pair differs in substance, not in format:

- ``citation_corruption``: a mapped limitation cites a different passage of the input,
  one the examiner didn't cite for this claim
- ``unsupported_rejection``: the same rejection, but asserting disclosure with no passages
- ``false_rejection``: an allowable claim rejected under 102 over the main cited
  reference, with every limitation "disclosed" by its best-matching passage
- ``overconfident_support``: a partially supported rejection presented as fully
  supported, with invented mappings for the unmapped limitations
- ``false_allowance``: a rejected claim allowed, with every limitation "not disclosed"
  (the counterweight to ``false_rejection``, so DPO doesn't just learn to reject)

Rejected outputs are templates, never treated as text anyone wrote.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from patent_reasoning.data.claim_records import word_coverage
from patent_reasoning.schema import (
    PRIOR_ART_STATUTES,
    Example,
    Ground,
    LimitationMapping,
    ModelOutput,
)

PAIR_TYPES = (
    "citation_corruption",
    "unsupported_rejection",
    "false_rejection",
    "overconfident_support",
    "false_allowance",
)


@dataclass
class Pair:
    pair_id: str
    example_id: str
    split: str
    type: str
    example: Example
    rejected: ModelOutput

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "example_id": self.example_id,
            "split": self.split,
            "type": self.type,
            "input": self.example.input.model_dump(),
            "chosen": self.example.target.model_dump(),
            "rejected": self.rejected.model_dump(),
        }


def _pick(options: list[str], key: str) -> str:
    """Deterministic choice keyed by a string."""
    return min(options, key=lambda o: hashlib.sha256(f"{key}:{o}".encode()).hexdigest())


def _passages(example: Example) -> dict[str, list[tuple[str, str]]]:
    """Document ID -> [(passage ID, text)] of the input."""
    return {d.document_id: [(p.passage_id, p.text) for p in d.passages]
            for d in example.input.prior_art if d.passages}  # fmt: skip


def _best_passage(limitation: str, passages: dict[str, list[tuple[str, str]]]) -> tuple[str, str]:
    """(document ID, passage ID) of the passage that best covers the limitation's words."""
    scored = [(word_coverage(limitation, text), doc, pid)
              for doc, ps in passages.items() for pid, text in ps]  # fmt: skip
    _, doc, pid = max(scored, key=lambda s: (s[0], s[2]))
    return doc, pid


def _has_art(example: Example) -> bool:
    return any(g.statute in PRIOR_ART_STATUTES for g in example.target.grounds)


def citation_corruption(example: Example) -> ModelOutput | None:
    target = example.target
    passages = _passages(example)
    cited = {p for m in target.claim_mapping for p in m.passage_ids}
    mapped = [m for m in target.claim_mapping if m.passage_ids]
    if not _has_art(example) or not mapped:
        return None
    victim = mapped[0]
    same_doc = [pid for pid, _ in passages.get(victim.document_id, []) if pid not in cited]
    others = [(doc, pid) for doc, ps in passages.items() for pid, _ in ps if pid not in cited]
    if same_doc:
        doc, pid = victim.document_id, _pick(same_doc, example.example_id)
    elif others:
        doc, pid = others[0]
    else:
        return None
    corrupted = victim.model_copy(update={"document_id": doc, "passage_ids": [pid]})
    mapping = [corrupted if m is victim else m for m in target.claim_mapping]
    return target.model_copy(update={"claim_mapping": mapping})


def unsupported_rejection(example: Example) -> ModelOutput | None:
    target = example.target
    if not _has_art(example) or not any(m.passage_ids for m in target.claim_mapping):
        return None
    mapping = [m.model_copy(update={"passage_ids": []}) for m in target.claim_mapping]
    return target.model_copy(update={"claim_mapping": mapping, "evidence_status": "supported"})


def false_rejection(example: Example, limitations: list[str]) -> ModelOutput | None:
    if example.target.grounds or not limitations:
        return None
    passages = _passages(example)
    if not passages:
        return None
    main = max(passages, key=lambda d: (len(passages[d]), d))
    mapping, locators = [], []
    for lim in limitations:
        _, pid = _best_passage(lim, {main: passages[main]})
        mapping.append(LimitationMapping(limitation=lim, disclosed=True, document_id=main,
                                         passage_ids=[pid]))  # fmt: skip
        locators.append(pid.split(":", 1)[1])
    n = example.input.claim.number
    explanation = (f"Claim {n} is anticipated by {main}, which discloses each limitation "
                   f"(passages {', '.join(locators)}).")  # fmt: skip
    return ModelOutput(grounds=[Ground(statute="102", references=[main])], claim_mapping=mapping,
                       explanation=explanation, evidence_status="supported")  # fmt: skip


def overconfident_support(example: Example, limitations: list[str]) -> ModelOutput | None:
    target = example.target
    if not _has_art(example) or target.evidence_status != "partially_supported":
        return None
    passages = _passages(example)
    mapped = {m.limitation for m in target.claim_mapping}
    missing = [lim for lim in limitations if lim not in mapped]
    if not missing or not passages:
        return None
    invented = []
    for lim in missing:
        doc, pid = _best_passage(lim, passages)
        invented.append(LimitationMapping(limitation=lim, disclosed=True, document_id=doc,
                                          passage_ids=[pid]))  # fmt: skip
    return target.model_copy(update={"claim_mapping": list(target.claim_mapping) + invented,
                                     "evidence_status": "supported"})  # fmt: skip


def false_allowance(example: Example, limitations: list[str]) -> ModelOutput | None:
    if not _has_art(example) or not limitations:
        return None
    n = example.input.claim.number
    return ModelOutput(
        grounds=[],
        claim_mapping=[LimitationMapping(limitation=lim, disclosed=False) for lim in limitations],
        explanation=f"The cited art does not disclose or suggest the limitations of claim {n}.",
        evidence_status="not_applicable",
    )


def grounded_in_input(example: Example, output: ModelOutput) -> bool:
    docs, passages = example.input.document_ids(), example.input.passage_ids()
    refs = {r for g in output.grounds for r in g.references}
    used = {p for m in output.claim_mapping for p in m.passage_ids}
    mapped_docs = {m.document_id for m in output.claim_mapping if m.document_id}
    return refs <= docs and mapped_docs <= docs and used <= passages


def make_pairs(example: Example, split: str, limitations: list[str]) -> list[Pair]:
    """All applicable pairs for one example. ``limitations`` is the claim's own limitations
    (from its claim record), which the target's mapping may cover only partly."""
    builders = {
        "citation_corruption": lambda: citation_corruption(example),
        "unsupported_rejection": lambda: unsupported_rejection(example),
        "false_rejection": lambda: false_rejection(example, limitations),
        "overconfident_support": lambda: overconfident_support(example, limitations),
        "false_allowance": lambda: false_allowance(example, limitations),
    }
    pairs = []
    for kind, build in builders.items():
        rejected = build()
        if rejected is None or rejected == example.target:
            continue
        if not grounded_in_input(example, rejected):
            raise ValueError(f"{kind} for {example.example_id} cites outside the input")
        pairs.append(Pair(f"{example.example_id}:{kind}", example.example_id, split, kind,
                          example, rejected))  # fmt: skip
    return pairs
