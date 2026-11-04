"""Audit of the split dataset: automatic checks on every example, and a stratified sample
rendered for manual review.

Automatic checks find problems a rule can see (a limitation that isn't in the claim, a
dependent claim without its parent, a mapped passage that shares no words with its
limitation). The manual review answers what a rule can't: whether the label, the claim
text, the mapping and the explanation are what the examiner actually said about this claim.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from patent_reasoning.data.claim_records import word_coverage
from patent_reasoning.data.office_action import Block, parse_claim_list, sentence_end
from patent_reasoning.schema import Example

CHECKS = (
    "limitation_not_in_claim",
    "parent_claim_missing",
    "mapped_passage_no_overlap",
    "explanation_missing",
    "examiner_text_missing",
    "provenance_incomplete",
    "conflicting_duplicate",
)

# Verdict fields of the manual review and their allowed values.
RUBRIC = {
    "decision": ("correct", "wrong", "unclear"),
    "claim_text": ("ok", "wrong", "unclear"),
    "mapping": ("ok", "partly", "wrong", "n/a"),
    "explanation": ("ok", "off_claim", "wrong", "n/a"),
    "leak": ("none", "label_in_input"),
}


def final(verdict: dict) -> dict:
    """The verdict with any adjudicated fields applied."""
    return verdict | {k: v for k, v in verdict.get("adjudication", {}).items() if k in RUBRIC}


def is_error(verdict: dict) -> bool:
    """Whether the (final) verdict finds the target wrong somewhere. "partly" mappings and
    "unclear" fields aren't errors: the first is flagged in provenance, the second unproven."""
    v = final(verdict)
    return (v["decision"] == "wrong" or v["claim_text"] == "wrong" or v["mapping"] == "wrong"
            or v["explanation"] in ("off_claim", "wrong") or v["leak"] != "none")  # fmt: skip


_DEPENDS = re.compile(r"\b(?:of|in|to|by|with|from)\s+claims?\s+(\d+)", re.IGNORECASE)
_SPACES = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _SPACES.sub(" ", text).strip().lower()


def stratum(example: Example) -> str:
    """Decision class for sampling: the statutes joined, or "no_rejection"."""
    return "+".join(sorted(example.target.decision))


def check(example: Example) -> list[str]:
    """Problems with one example that a rule can see."""
    issues = []
    claim = _norm(example.input.claim.text)
    target = example.target
    if any(_norm(m.limitation) not in claim for m in target.claim_mapping):
        issues.append("limitation_not_in_claim")
    # A claim naming itself ("The station of claim 11" in claim 11) is a drafting error
    # with no parent to supply.
    named = {int(n) for n in _DEPENDS.findall(example.input.claim.text[:200])}
    named.discard(example.input.claim.number)
    if named - {c.number for c in example.input.parent_claims}:
        issues.append("parent_claim_missing")
    passages = {p.passage_id: p.text for d in example.input.prior_art for p in d.passages}
    if any(m.disclosed and m.passage_ids
           and max(word_coverage(m.limitation, passages[p]) for p in m.passage_ids) == 0
           for m in target.claim_mapping):  # fmt: skip
        issues.append("mapped_passage_no_overlap")
    if target.grounds and target.explanation is None:
        issues.append("explanation_missing")
    prov = example.provenance
    if target.grounds and not prov.examiner_text:
        issues.append("examiner_text_missing")
    if not (prov.application_id and prov.office_action_id and prov.mail_date and prov.art_unit):
        issues.append("provenance_incomplete")
    return issues


def conflicting_duplicates(examples: list[Example]) -> set[str]:
    """Examples whose claim and prior art match another example's, but whose decision
    differs. The model couldn't tell them apart, so at least one label is unlearnable."""
    by_input: dict[str, list[Example]] = defaultdict(list)
    for e in examples:
        key = _norm(" ".join(c.text for c in [e.input.claim, *e.input.parent_claims]))
        key += "|" + ",".join(sorted(e.input.document_ids()))
        by_input[hashlib.sha256(key.encode()).hexdigest()].append(e)
    out = set()
    for group in by_input.values():
        if len({e.target.decision for e in group}) > 1:
            out.update(e.example_id for e in group)
    return out


def sample(examples: dict[str, list[Example]], quotas: dict[str, int], salt: str) -> list[Example]:
    """Per split, ``quotas[split]`` examples taken round-robin over decision classes, each
    class in salted-hash order, so rare classes are represented."""

    def order(e: Example) -> str:
        return hashlib.sha256(f"{salt}:{e.example_id}".encode()).hexdigest()

    out = []
    for split, n in quotas.items():
        strata: dict[str, list[Example]] = defaultdict(list)
        for e in sorted(examples.get(split, []), key=order):
            strata[stratum(e)].append(e)
        queues = [strata[k] for k in sorted(strata)]
        taken = 0
        while taken < n and any(queues):
            for q in queues:
                if q and taken < n:
                    out.append(q.pop(0))
                    taken += 1
    return out


_DISPOSITION = re.compile(r"rejected under|objected to|allow|withdrawn|cancel", re.IGNORECASE)
_CLAIMS = re.compile(r"Claims?\s+(\d[\d,;\s\-–&and/through]*)", re.IGNORECASE)


def disposition_sentences(blocks: list[Block], number: int, limit: int = 12) -> list[str]:
    """Sentences of the office action that dispose of claims and name claim ``number``
    (directly or in a range), so a reviewer can check the label against the action."""
    out = []
    for block in blocks:
        text, start = block.text, 0
        while start < len(text) and len(out) < limit:
            end = sentence_end(text, start)
            sentence = text[start:end].strip()
            start = end
            if not _DISPOSITION.search(sentence):
                continue
            named = {c for m in _CLAIMS.finditer(sentence) for c in parse_claim_list(m.group(1))}
            if number in named:
                out.append(sentence[:600])
    return out


def _clip(text: str, n: int) -> str:
    text = _SPACES.sub(" ", text).strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def render(example: Example, split: str, issues: list[str], dispositions: list[str]) -> str:
    """One example as Markdown for the manual review."""
    t, prov = example.target, example.provenance
    passages = {p.passage_id: p.text for d in example.input.prior_art for p in d.passages}
    lines = [
        f"## {example.example_id}",
        f"- split `{split}`, decision `{stratum(example)}`, source `{prov.decision_source}`, "
        f"art unit {prov.art_unit}, mailed {prov.mail_date}",
        f"- claims from `{prov.claims_source}`, amended before action "
        f"{prov.claims_amended_before_action}, verified {prov.claim_text_verified}",
        f"- flags: {', '.join(prov.flags) or 'none'}; automatic checks: "
        f"{', '.join(issues) or 'none'}",
        "",
        f"**Claim.** {example.input.claim.text}",
    ]
    for parent in example.input.parent_claims:
        lines.append(f"\n**Parent claim {parent.number}.** {_clip(parent.text, 900)}")
    docs = [f"{d.document_id} ({d.kind}, {len(d.passages)} passages)"
            for d in example.input.prior_art]  # fmt: skip
    lines += ["", f"**Prior art in input.** {'; '.join(docs)}", "", "**Target.**"]
    grounds = [f"{g.statute} over {', '.join(g.references) or '-'}" for g in t.grounds]
    lines.append(f"- grounds: {'; '.join(grounds) or 'none (no rejection)'}")
    lines.append(f"- evidence status: {t.evidence_status}")
    for m in t.claim_mapping:
        lines.append(f"- limitation: \"{_clip(m.limitation, 300)}\" → disclosed={m.disclosed}, "
                     f"{m.document_id or '-'} {m.passage_ids or ''}")  # fmt: skip
        for pid in m.passage_ids:
            lines.append(f"    - `{pid}`: {_clip(passages[pid], 700)}")
    lines.append(f"- explanation: {_clip(t.explanation or '(none)', 1500)}")
    lines += ["", f"**Examiner text (provenance).** {_clip(prov.examiner_text, 2000)}"]
    cites = [f"{c.raw} → {c.resolved_passage_ids}" for c in prov.pin_cites]
    lines.append(f"\n**Pin cites.** {'; '.join(cites) or 'none'}")
    lines.append("\n**Office action sentences naming this claim.**")
    lines += [f"- {s}" for s in dispositions] or ["- (none found)"]
    return "\n".join(lines) + "\n"
