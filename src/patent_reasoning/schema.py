"""Data contracts for claim-level examination examples.

An ``Example`` has three parts:

- ``input``: what the model sees, which is one claim (plus the claims it depends on) and
  the cited prior art as passages. It never includes the office action, which states the
  outcome.
- ``target``: the reference output, with the same shape as the model's output
  (``ModelOutput``).
- ``provenance``: where every field came from and how much to trust it. Never shown to
  the model.

The JSON Schemas in ``docs/schemas/`` are exported from these models by
``scripts/export_schemas.py``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.2"

Statute = Literal["101", "102", "103", "112"]
PRIOR_ART_STATUTES = ("102", "103")
NO_REJECTION = "no_rejection"
DECISION_CLASSES = (NO_REJECTION, "102", "103", "112", "101")

EvidenceStatus = Literal["supported", "partially_supported", "unsupported", "not_applicable"]
DocumentKind = Literal["us_publication", "us_patent", "foreign", "non_patent"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class ClaimText(_Strict):
    number: int = Field(ge=1)
    text: str = Field(min_length=1)


class Passage(_Strict):
    passage_id: str = Field(
        description="Unique within the example, conventionally '<document_id>:<locator key>', "
        "e.g. '20130173955:0092'."
    )
    locator: str = Field(description="How the document cites it, e.g. '[0092]'.")
    text: str = Field(min_length=1)


class PriorArtDocument(_Strict):
    document_id: str = Field(
        description="US publication (11 digits), US patent (number without commas), or "
        "foreign/non-patent identifier, as normalized by the office action parser."
    )
    kind: DocumentKind
    title: str = ""
    passages: list[Passage] = Field(
        default_factory=list,
        description="Retrieved passages. Empty means the document is known only by its ID.",
    )


class ExampleInput(_Strict):
    claim: ClaimText
    parent_claims: list[ClaimText] = Field(
        default_factory=list,
        description="Claims this claim depends on, nearest first, so the full scope is visible.",
    )
    prior_art: list[PriorArtDocument] = Field(
        default_factory=list, description="The art cited in the office action."
    )

    def passage_ids(self) -> set[str]:
        return {p.passage_id for d in self.prior_art for p in d.passages}

    def document_ids(self) -> set[str]:
        return {d.document_id for d in self.prior_art}


# ---------------------------------------------------------------------------
# Target / model output
# ---------------------------------------------------------------------------


class Ground(_Strict):
    statute: Statute
    references: list[str] = Field(
        default_factory=list,
        description="Document IDs the ground relies on. 102 needs exactly one, 103 at least "
        "one, and 101/112 none.",
    )

    @model_validator(mode="after")
    def _references_match_statute(self) -> Ground:
        n = len(self.references)
        if self.statute == "102" and n != 1:
            raise ValueError("a 102 ground relies on exactly one reference")
        if self.statute == "103" and n < 1:
            raise ValueError("a 103 ground needs at least one reference")
        if self.statute in ("101", "112") and n:
            raise ValueError(f"a {self.statute} ground does not cite prior art")
        return self


class LimitationMapping(_Strict):
    limitation: str = Field(min_length=1, description="A limitation of this claim, verbatim.")
    disclosed: bool = Field(description="Whether the cited art discloses or suggests it.")
    document_id: str | None = None
    passage_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _disclosure_has_a_source(self) -> LimitationMapping:
        if self.disclosed and self.document_id is None:
            raise ValueError("a disclosed limitation must name the document that discloses it")
        if self.passage_ids and self.document_id is None:
            raise ValueError("passages need a document_id")
        return self


class ModelOutput(_Strict):
    """What the model produces for one claim, and the shape of every reference target."""

    grounds: list[Ground] = Field(
        default_factory=list, description="Rejection grounds. Empty means no rejection."
    )
    claim_mapping: list[LimitationMapping] = Field(
        default_factory=list,
        description="This claim's own limitations (parent limitations belong to the parent "
        "claim's example).",
    )
    explanation: str | None = Field(
        default=None,
        description="Reasoning for the decision. Null in a target when the examiner gave none.",
    )
    evidence_status: EvidenceStatus

    @model_validator(mode="after")
    def _status_matches_decision(self) -> ModelOutput:
        has_art = any(g.statute in PRIOR_ART_STATUTES for g in self.grounds)
        if has_art and self.evidence_status == "not_applicable":
            raise ValueError("a 102/103 rejection needs an evidence status")
        if not has_art and self.evidence_status != "not_applicable":
            raise ValueError("evidence status applies only to 102/103 rejections")
        return self

    @property
    def decision(self) -> frozenset[str]:
        """Decision labels for scoring: the set of statutes, or {"no_rejection"}."""
        return frozenset(g.statute for g in self.grounds) or frozenset({NO_REJECTION})


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class PinCite(_Strict):
    raw: str = Field(description="As written by the examiner, e.g. 'paragraph [0092]'.")
    kind: Literal["paragraph", "column_line"]
    document_id: str | None
    resolved_passage_ids: list[str] = Field(default_factory=list)


class Provenance(_Strict):
    application_id: str
    office_action_id: str = Field(description="File-wrapper document ID (= OARD ifw_number).")
    document_code: str = Field(description="CTNF (non-final) or CTFR (final).")
    mail_date: str
    art_unit: str = ""
    uspc_class: str = ""

    decision_source: Literal["rejection_statement", "allowable_dependent", "allowed"] = Field(
        description="Where the decision comes from in the office action text."
    )
    oard_agrees: bool | None = Field(
        default=None,
        description="Whether the OARD rows give the same statutes for this claim; None if "
        "OARD has no row covering it.",
    )
    claims_source: Literal["publication", "as_filed_ocr"]
    claims_amended_before_action: bool
    claim_text_verified: bool | None = Field(
        default=None,
        description="Whether the examiner's restatement of the claim matches the input claim "
        "text; None if not checked.",
    )

    evidence_condition: Literal["action_cited", "retrieved"] = Field(
        default="action_cited",
        description="How the input passages were chosen (see docs/SPEC.md, Evidence).",
    )

    examiner_text: str = Field(
        default="",
        description="The examiner's statement and reasoning for this claim, verbatim. "
        "Reference material only; never model input.",
    )
    pin_cites: list[PinCite] = Field(default_factory=list)
    explanation_source: Literal["examiner_reasoning", "reasons_for_allowance", "none"] = Field(
        default="none",
        description="examiner_reasoning: the examiner's discussion of this claim in a "
        "rejection; reasons_for_allowance: the examiner's statement of reasons for allowable "
        "subject matter that names this claim.",
    )
    flags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


class Example(_Strict):
    schema_version: Literal["1.2"] = SCHEMA_VERSION
    example_id: str = Field(description="'<office_action_id>:<claim number>'.")
    input: ExampleInput
    target: ModelOutput
    provenance: Provenance

    @model_validator(mode="after")
    def _consistent(self) -> Example:
        expected_id = f"{self.provenance.office_action_id}:{self.input.claim.number}"
        if self.example_id != expected_id:
            raise ValueError(f"example_id should be {expected_id!r}")
        docs, passages = self.input.document_ids(), self.input.passage_ids()
        for ground in self.target.grounds:
            if missing := set(ground.references) - docs:
                raise ValueError(f"ground cites documents not in the input: {sorted(missing)}")
        for m in self.target.claim_mapping:
            if m.document_id is not None and m.document_id not in docs:
                raise ValueError(f"mapping cites a document not in the input: {m.document_id}")
            if missing := set(m.passage_ids) - passages:
                raise ValueError(f"mapping cites passages not in the input: {sorted(missing)}")
        if self.provenance.decision_source != "rejection_statement" and self.target.grounds:
            raise ValueError("an allowed or allowable claim can't have rejection grounds")
        return self
