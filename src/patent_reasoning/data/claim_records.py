"""Claim-level records: one per claim per office action.

For each office action this finds what the examiner decided about every claim, from the
action's own text:

- rejection statements give the grounds (statute plus the references the statement names)
- "objected to as being dependent upon a rejected base claim, but would be allowable" and
  "are allowed" give the no-rejection claims

Each claim gets its text and the claims it depends on, its limitations (split from the
claim text), the examiner's reasoning for that claim, and the examiner's pin cites mapped
to the limitations they support. A claim-restatement check and the amendment's claim
status labels say whether the claim text we have is the text the examiner examined.

These records are an intermediate stage; the model-facing ``Example`` is built from them
together with the retrieved prior-art passages.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import duckdb

from patent_reasoning.data.evidence import normalize_document_id
from patent_reasoning.data.joined import (
    Joiner,
    claims_amended_before,
    fulltext_uri,
    latest_claims_listing,
    xml_download_url,
)
from patent_reasoning.data.oard import parse_claim_numbers
from patent_reasoning.data.office_action import (
    SELF,
    Block,
    PinCite,
    RejectionText,
    attribute_pin_cites,
    extract_rejections,
    find_aliases,
    find_references,
    is_heading,
    parse_blocks,
    parse_claim_list,
    sentence_end,
)
from patent_reasoning.data.publications import (
    Claim,
    Publication,
    parse_claim_statuses,
    parse_publication,
)

STATUTES = ("101", "102", "103", "112")
PRIOR_ART_STATUTES = ("102", "103")

# Restatement overlap at or above this means the examiner restated the claim text we have.
RESTATEMENT_THRESHOLD = 0.5
# A limitation is matched to a pin cite when this share of its content words appears in the
# examiner's text just before the cite.
MAPPING_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Dispositions
# ---------------------------------------------------------------------------

_LIST = r"(?P<claims>\d+(?:\s*(?:,|;|-|–|&|and|or|through|to)\s*(?:claims?\s+)?\d+)*)"
_SUBJECT = rf"Claims?(?:\(s\))?\s+{_LIST},?\s+(?:is/are|is|are)\s+"
_ALLOWABLE_DEPENDENT = re.compile(
    _SUBJECT + r"(?:also\s+)?objected\s+to\s+as\s+being\s+dependent\s+(?:up)?on\s+a\s+rejected"
    r"\s+base\s+claims?,?\s+but\s+would\s+be\s+allowable",
    re.IGNORECASE,
)
_ALLOWED = re.compile(_SUBJECT + r"allowed\b", re.IGNORECASE)
# 07-43-02: "Claim 13 would be allowable if rewritten to overcome the rejection(s) under 35
# U.S.C. 112 ..." The claim is allowable over the art but still rejected under 112.
_ALLOWABLE_OVER_ART = re.compile(
    rf"Claims?(?:\(s\))?\s+{_LIST},?\s+(?:is/are\s+|is\s+|are\s+)?(?:would\s+be|is|are)\s+"
    r"allowable\s+if\s+rewritten\s+to\s+overcome",
    re.IGNORECASE,
)
_WITHDRAWN = re.compile(_SUBJECT + r"withdrawn\b", re.IGNORECASE)
_REASONS = re.compile(r"statement of reasons for (?:the indication of allowable subject matter|"
                      r"allowance)|reasons for allowance", re.IGNORECASE)  # fmt: skip


@dataclass
class Dispositions:
    allowable_dependent: set[int] = field(default_factory=set)
    allowed: set[int] = field(default_factory=set)
    allowable_over_art: set[int] = field(default_factory=set)
    withdrawn: set[int] = field(default_factory=set)
    reasons_for_allowance: list[str] = field(default_factory=list)


def parse_dispositions(blocks: list[Block]) -> Dispositions:
    out = Dispositions()
    targets = (
        (_ALLOWABLE_DEPENDENT, out.allowable_dependent),
        (_ALLOWED, out.allowed),
        (_ALLOWABLE_OVER_ART, out.allowable_over_art),
        (_WITHDRAWN, out.withdrawn),
    )
    reasons: str | None = None
    for block in blocks:
        for pattern, claims in targets:
            for m in pattern.finditer(block.text):
                claims.update(parse_claim_list(m.group("claims")))
        if _REASONS.search(block.text[:200]):
            if reasons:
                out.reasons_for_allowance.append(reasons)
            reasons = block.text
        elif reasons is not None and (is_heading(block) or block.fp_number):
            out.reasons_for_allowance.append(reasons)
            reasons = None
        elif reasons is not None:
            reasons += "\n" + block.text
    if reasons:
        out.reasons_for_allowance.append(reasons)
    return out


def claims_named(text: str) -> set[int]:
    return {c for m in re.finditer(rf"\bclaims?\s+{_LIST}", text, re.IGNORECASE)
            for c in parse_claim_list(m.group("claims"))}  # fmt: skip


# ---------------------------------------------------------------------------
# Grounds
# ---------------------------------------------------------------------------


@dataclass
class GroundRecord:
    statute: str
    references: list[str]
    statement: str
    valid: bool  # reference count fits the statute (102: one, 103: at least one, else none)


def _introduces(text: str, name_start: int, name_end: int, number_at: int) -> bool:
    """Whether the name at text[name_start:name_end] labels the number at ``number_at``:
    close by, with no "in view of", "and", "or" or ";" between them."""
    between = text[name_end:number_at] if number_at >= name_end else text[number_at:name_start]
    return len(between) < 50 and not re.search(r"view\s+of|\band\b|\bor\b|;", between)


def statement_references(statement: str, aliases: dict[str, str]) -> list[str]:
    """Documents a rejection statement relies on, in order of first mention.

    An explicit number beats a shorthand name next to it ("Bates et al. (US 7,343,588)"
    means that patent even if "Bates" named another document earlier in the action), and
    priority applications ("claims priority to EP 05001407.5") aren't references.
    """
    explicit = [
        (o, d)
        for o, d in find_references(statement)
        if not re.search(r"priority", statement[max(0, o - 60) : o], re.IGNORECASE)
    ]
    mentions = list(explicit)
    for name, doc_id in aliases.items():
        m = re.search(rf"\b{re.escape(name)}\b", statement, re.IGNORECASE)
        if m and not any(_introduces(statement, m.start(), m.end(), o) for o, _ in explicit):
            mentions.append((m.start(), doc_id))
    seen: list[str] = []
    for _, doc_id in sorted(mentions):
        if doc_id not in seen and doc_id != SELF:
            seen.append(doc_id)
    return seen


def expected_references(statement: str, statute: str) -> int:
    """How many references the statement's wording implies: "A in view of B and further in
    view of C" is three."""
    if statute == "102":
        return 1
    return 1 + len(re.findall(r"\bview\s+of\s+(?!In\s+re\b)", statement, re.IGNORECASE))


# A journal article or other non-patent document: a quoted title or journal details.
_NON_PATENT = re.compile(r"[“\"][^”\"]{12,}[”\"]|\bJournal\b|\bVol\.\s*\d|\bpp?\.\s*\d", re.I)


_ALTERNATIVE_103 = re.compile(
    r"in\s+the\s+alternative,?\s+(?:under\s+)?(?:pre-AIA\s+)?35\s*U\.?\s*S\.?\s*C\.?\s*§*\s*103",
    re.IGNORECASE,
)


def make_grounds(rej: RejectionText, aliases: dict[str, str]) -> list[GroundRecord]:
    """The grounds in one rejection statement.

    "Anticipated by or, in the alternative, under 35 U.S.C. 103(a) as obvious over X" is
    two grounds over the same reference. A 102/103 ground is valid when it resolves the
    references its wording implies (102: exactly one).
    """
    refs = statement_references(rej.statement, aliases) if rej.statute in PRIOR_ART_STATUTES else []
    statutes = [rej.statute]
    if rej.statute == "102" and _ALTERNATIVE_103.search(rej.statement):
        statutes.append("103")
    grounds = []
    for statute in statutes:
        valid = True
        if statute == "102":
            valid = len(refs) == 1
        elif statute == "103":
            valid = len(refs) >= expected_references(rej.statement, "103")
        grounds.append(GroundRecord(statute, refs, rej.statement, valid))
    return grounds


# ---------------------------------------------------------------------------
# Claim text
# ---------------------------------------------------------------------------

_NUMBER_PREFIX = re.compile(r"^\s*\d+\s*(?:\([^)]{0,40}\))?\s*[.:]?\s*")
_DEPENDENCY = re.compile(
    r"\b(?:of|to|in|by|with|from|under)\s+(?:any\s+(?:one\s+)?of\s+)?claims?\s+(\d+)", re.I
)
_DEPENDENT_PREAMBLE = re.compile(
    r"^.{0,200}?\bclaims?\s+\d+(?:\s*(?:,|or|and|-|to|through)\s*(?:claims?\s+)?\d+)*"
    r"(?:\s*,?\s*(?:above|herein)?)\s*,?\s*"
    r"(?:(?:further\s+)?(?:comprising|including|comprises|includes|wherein|characteri[sz]ed"
    r"(?:\s+in\s+that|\s+by)?|in\s+which|where|whereby)\b\s*:?\s*)?",
    re.IGNORECASE | re.DOTALL,
)
_TRANSITION = re.compile(
    r"\b(?:comprising|consisting\s+essentially\s+of|consisting\s+of|including|comprises|"
    r"which\s+comprises|characteri[sz]ed\s+(?:in\s+that|by)|having)\b\s*:?\s*",
    re.IGNORECASE,
)
_LIMITATION_SPLIT = re.compile(r";\s*(?:and\s+|or\s+)?|,\s+(?=wherein\b)", re.IGNORECASE)


def claim_body(text: str) -> str:
    """Claim text without its leading number ("2. ", "22 (original). ")."""
    return _NUMBER_PREFIX.sub("", text, count=1)


def dependencies(claim: Claim) -> list[int]:
    """Direct parents. OCR'd claims have no claim-ref markup, so fall back to the text."""
    if claim.depends_on:
        return claim.depends_on
    m = _DEPENDENCY.search(claim_body(claim.text)[:250])
    return [int(m.group(1))] if m and int(m.group(1)) < claim.number else []


def ancestors(number: int, claims: dict[int, Claim]) -> list[int]:
    """All claims ``number`` depends on, nearest first."""
    out: list[int] = []
    queue = list(dependencies(claims[number])) if number in claims else []
    while queue:
        parent = queue.pop(0)
        if parent in out or parent == number or parent not in claims:
            continue
        out.append(parent)
        queue += dependencies(claims[parent])
    return out


def split_limitations(text: str, dependent: bool) -> list[str]:
    """Split a claim into its own limitations, each a verbatim substring of ``text``.

    Independent claims drop the preamble (up to "comprising" or a similar transition) and
    split the body at semicolons and before ", wherein". Dependent claims drop "The method
    of claim 1, further comprising" and split the rest the same way.
    """
    body = claim_body(text)
    if dependent:
        m = _DEPENDENT_PREAMBLE.match(body)
        rest = body[m.end() :] if m else body
    else:
        m = _TRANSITION.search(body)
        rest = body[m.end() :] if m else body
    parts = _LIMITATION_SPLIT.split(rest)
    limitations = []
    for part in parts:
        part = part.strip().strip(".,;: ").strip()
        part = re.sub(r"^(?:and|or)\s+", "", part)
        if len(re.findall(r"[A-Za-z]{3,}", part)) >= 2:
            limitations.append(part)
    return limitations


# ---------------------------------------------------------------------------
# Per-claim examiner reasoning
# ---------------------------------------------------------------------------

_MARKER = re.compile(
    r"(?:Regarding|As\s+to|As\s+per|As\s+for|With\s+respect\s+to|With\s+regard\s+to|"
    r"Referring\s+to|In\s+regard\s+to|In\s+re|Re:?|Considering|Concerning|For)\s+"
    r"(?:the\s+)?(?:independent\s+|dependent\s+)?claims?\s+"
    r"(?P<claims>\d+(?:\s*(?:,|-|–|&|and|or|through|to)\s*\d+)*)",
    re.IGNORECASE,
)


def _at_sentence_start(text: str, pos: int) -> bool:
    before = text[:pos].rstrip()
    return not before or before[-1] in ".:;)\n" or bool(re.search(r"(?:^|\s)\d+\.$", before))


@dataclass
class Segment:
    claims: list[int]
    start: int
    end: int


def claim_segments(rej: RejectionText) -> list[Segment]:
    """Split a rejection's reasoning at "Regarding claim N" style markers.

    The text before the first marker (after the statement) is attributed to the first
    claim of the statement when that claim has no marker of its own; examiners usually
    discuss the independent claim first without announcing it.
    """
    text = rej.reasoning
    markers = [m for m in _MARKER.finditer(text) if _at_sentence_start(text, m.start())]
    segments = []
    for i, m in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        segments.append(Segment(parse_claim_list(m.group("claims")), m.start(), end))
    first_marker = markers[0].start() if markers else len(text)
    covered = {c for s in segments for c in s.claims}
    lead_start = len(rej.statement) if text.startswith(rej.statement) else 0
    if rej.claims and rej.claims[0] not in covered and first_marker - lead_start > 100:
        segments.insert(0, Segment([rej.claims[0]], lead_start, first_marker))
    return segments


def sentences(text: str, start: int = 0) -> list[tuple[int, int]]:
    spans = []
    while start < len(text):
        end = sentence_end(text, start)
        spans.append((start, end))
        start = end
    return spans


def claim_windows(rej: RejectionText, number: int) -> tuple[list[Segment], str]:
    """The parts of a rejection's reasoning about claim ``number``, and how they were found.

    - "marker": "Regarding claim N" style segments
    - "mention": sentences that name the claim ("[as in claims 3-4]", "addresses claim 19")
    - "shared": a 101/112 rejection's whole reasoning, which usually argues the claim group
      as a whole
    - "none"
    """
    if found := [s for s in claim_segments(rej) if number in s.claims]:
        return found, "marker"
    text = rej.reasoning
    lead = len(rej.statement) if text.startswith(rej.statement) else 0
    found = [Segment([number], a, b) for a, b in sentences(text, lead)
             if number in claims_named(text[a:b])]  # fmt: skip
    if found:
        return found, "mention"
    if rej.statute in ("101", "112") and len(text) - lead > 50:
        return [Segment(rej.claims, lead, len(text))], "shared"
    return [], "none"


# ---------------------------------------------------------------------------
# Text overlap
# ---------------------------------------------------------------------------

_STOPWORDS = set(
    "the a an of to and or in on for with by at from as is are be said that which wherein "
    "each said one least further comprising including having".split()
)
_BRACKETS = re.compile(r"\[[^\]]{0,80}\]|\([^)]{0,120}\)")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _BRACKETS.sub(" ", text.lower()))


def ngram_overlap(claim_text: str, examiner_text: str, n: int = 4) -> float | None:
    """Share of the claim's word n-grams that the examiner's text repeats."""
    a, b = _tokens(claim_text), _tokens(examiner_text)
    grams = {tuple(a[i : i + n]) for i in range(len(a) - n + 1)}
    if not grams:
        return None
    other = {tuple(b[i : i + n]) for i in range(len(b) - n + 1)}
    return len(grams & other) / len(grams)


def word_coverage(limitation: str, text: str) -> float:
    words = {w for w in _tokens(limitation) if w not in _STOPWORDS and len(w) > 2}
    if not words:
        return 0.0
    return len(words & set(_tokens(text))) / len(words)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class ExaminerEvidence:
    document_id: str | None
    kind: str  # "paragraph" or "column_line"
    raw: str
    start: str
    end: str
    paragraphs: list[str] = field(default_factory=list)  # resolved printed paragraph numbers
    limitations: list[int] = field(default_factory=list)  # indexes into ClaimRecord.limitations


@dataclass
class ClaimRecord:
    example_id: str
    application_id: str
    office_action_id: str
    document_code: str
    mail_date: str
    art_unit: str
    uspc_class: str
    claim_number: int
    claim_text: str
    parent_claims: list[int]  # nearest first
    parent_claim_texts: list[str]
    claims_source: str
    decision_source: str  # rejection_statement, allowable_dependent, allowed
    grounds: list[GroundRecord]
    decision: list[str]
    art_allowable: bool
    oard_statutes: list[str] | None
    oard_agrees: bool | None
    limitations: list[str]
    examiner_text: str
    explanation_source: str  # examiner_reasoning, reasons_for_allowance, none
    evidence: list[ExaminerEvidence]
    cited_documents: list[str]
    claims_amended_before_action: bool  # any amendment (incl. spec-only) before the action
    claims_listing_before_action: bool  # a claim listing was filed after the application
    pending_status: str | None
    restatement_overlap: float | None
    claim_text_verified: bool | None
    flags: list[str] = field(default_factory=list)

    @property
    def claim_text_reliable(self) -> bool:
        """Verified, or never at risk: any amendment of the claims has to come with a full
        claim listing (37 CFR 1.121(c)), so without a later listing the claims as filed
        are the claims examined."""
        return self.claim_text_verified is True or (
            self.claim_text_verified is None and not self.claims_listing_before_action
        )

    @property
    def quality(self) -> str:
        """ "grounded": trustworthy label and claim text, and for a 102/103 rejection at least
        one examiner pin cite resolved to a paragraph; "labeled": the same without the
        resolved evidence; "weak": anything else."""
        if not self.claim_text_reliable or "conflicting_disposition" in self.flags:
            return "weak"
        if any(not g.valid for g in self.grounds):
            return "weak"
        if "no_prior_art_text" in self.flags:
            return "labeled"  # nothing to judge the claim against
        has_art = any(g.statute in PRIOR_ART_STATUTES for g in self.grounds)
        if has_art and not any(e.paragraphs for e in self.evidence):
            return "labeled"
        return "grounded"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["claim_text_reliable"] = self.claim_text_reliable
        d["quality"] = self.quality
        return d


def verify_claim_text(
    overlap: float | None, pending_status: str | None
) -> tuple[bool | None, list[str]]:
    """Whether the claim text is what the examiner examined, plus explanatory flags.

    A status label from the latest amendment decides when present ("original" means the
    as-filed text still stands). Otherwise a close restatement in the examiner's reasoning
    confirms it. A loose restatement proves nothing on its own: examiners paraphrase.
    """
    flags = []
    if pending_status is not None:
        if pending_status == "original":
            return True, flags
        flags.append(f"claim_status_{pending_status.replace(' ', '_')}")
        return False, flags
    if overlap is not None and overlap >= RESTATEMENT_THRESHOLD:
        return True, flags
    if overlap is not None:
        flags.append("restatement_low_overlap")
    return None, flags


def map_evidence(
    limitations: list[str], text: str, start: int, end: int, cites: list[PinCite]
) -> list[tuple[PinCite, list[int]]]:
    """The pin cites in ``text[start:end]``, each with the limitations it supports.

    A limitation goes to the cite whose preceding examiner text (back to the previous cite)
    best restates it, if that covers at least ``MAPPING_THRESHOLD`` of its content words.
    """
    in_segment = sorted((c for c in cites if start <= c.offset < end), key=lambda c: c.offset)
    windows, prev = [], start
    for cite in in_segment:
        windows.append(text[prev : cite.offset])
        prev = cite.offset + len(cite.raw)
    hits: list[list[int]] = [[] for _ in in_segment]
    for i, lim in enumerate(limitations):
        scores = [word_coverage(lim, w) for w in windows]
        if scores and max(scores) >= MAPPING_THRESHOLD:
            hits[scores.index(max(scores))].append(i)
    return list(zip(in_segment, hits, strict=True))


def resolve_paragraphs(pub: Publication, start: str, end: str) -> list[str]:
    """Printed paragraph numbers in a publication that a "[start]-[end]" cite covers."""
    return [n for n in pub.paragraphs if start <= n <= end]


class ClaimRecordBuilder:
    def __init__(self, joiner: Joiner, con: duckdb.DuckDBPyConnection):
        self.joiner = joiner
        self.client = joiner.client
        self.con = con

    def _claims(self, docs, app) -> tuple[dict[int, Claim], str]:
        uri = fulltext_uri(app, "pgpub")
        if uri and (xml := self.client.fulltext_xml(uri)):
            if claims := parse_publication(xml).claims:
                return claims, "publication"
        if claims := self.joiner.as_filed_claims(docs, app):
            return claims, "as_filed_ocr"
        return {}, ""

    def build(self, action: dict[str, Any]) -> tuple[list[ClaimRecord], dict[str, Any]]:
        app_id, ifw = action["app_id"], action["ifw_number"]
        mail_date = str(action["mail_dt"])[:10]
        funnel: dict[str, Any] = {"office_action_id": ifw, "application_id": app_id}

        docs = self.client.documents(app_id)
        oa_doc = next((d for d in docs if d.get("documentIdentifier") == ifw), None)
        url = xml_download_url(oa_doc)
        oa_xml = self.client.document_xml(url) if url else None
        funnel["oa_text"] = bool(oa_xml)
        if not oa_xml:
            return [], funnel

        blocks = parse_blocks(oa_xml)
        rejections = extract_rejections(blocks)
        aliases = find_aliases("\n".join(b.text for b in blocks))
        for rej in rejections:
            attribute_pin_cites(rej, aliases)
        dispositions = parse_dispositions(blocks)

        app = self.client.application(app_id) or {}
        claims, claims_source = self._claims(docs, app)
        funnel["claims_source"] = claims_source
        amended = claims_amended_before(docs, app, mail_date)
        statuses: dict[int, str] = {}
        listing = latest_claims_listing(docs, app, mail_date)
        if listing and ((u := xml_download_url(listing)) and (xml := self.client.document_xml(u))):
            statuses = parse_claim_statuses(xml)
        funnel["claims_amended_before_action"] = amended
        funnel["claims_listing_before_action"] = listing is not None
        funnel["claim_statuses"] = bool(statuses)
        if not claims:
            return [], funnel

        if any(not g.valid for rej in rejections for g in make_grounds(rej, aliases)):
            aliases = self._with_inventor_aliases(aliases, app_id, ifw)
            funnel["inventor_aliases"] = True

        # Prior art cited anywhere in the action: the same documents for every claim. It
        # includes every reference a rejection statement relies on.
        oard_cited = {p for (p,) in self.con.execute(
            "SELECT DISTINCT parsed FROM citations WHERE ifw_number = ? AND citation_in_oa = '1' "
            "AND parsed IS NOT NULL", [ifw]).fetchall()}  # fmt: skip
        oard_cited = {n[0] for p in oard_cited if (n := normalize_document_id(p))}
        in_text = {d for rej in rejections for _, d in find_references(rej.reasoning)}
        relied_on = {d for rej in rejections for g in make_grounds(rej, aliases)
                     for d in g.references}  # fmt: skip
        cited = sorted((oard_cited | in_text | relied_on) - {SELF})
        pubs: dict[str, Publication | None] = {}
        for doc_id in cited:
            if doc_id.isdigit():
                pubs[doc_id] = self.joiner.fetch_reference(doc_id)[1]
        # Every passage the examiner pin-cited anywhere in the action (the evidence pool that
        # is shared by all of its claims).
        funnel["cited_passages"] = sorted({
            f"{c.document_id}:{n}"
            for rej in rejections for c in rej.pin_cites
            if c.kind == "paragraph" and len(c.document_id or "") == 11
            and (pub := pubs.get(c.document_id))
            for n in resolve_paragraphs(pub, c.start, c.end)
        })  # fmt: skip

        oard_rows = self.con.execute(
            "SELECT action_type, claim_numbers FROM rejections WHERE ifw_number = ?", [ifw]
        ).fetchall()
        oard_covered: dict[int, set[str]] = {}
        for action_type, numbers in oard_rows:
            for c in parse_claim_numbers(numbers) or []:
                oard_covered.setdefault(c, set())
                if action_type in STATUTES:
                    oard_covered[c].add(action_type)

        by_claim: dict[int, list[RejectionText]] = {}
        for rej in rejections:
            for c in rej.claims:
                by_claim.setdefault(c, []).append(rej)
        negatives = dispositions.allowable_dependent | dispositions.allowed
        candidates = (set(by_claim) | negatives) - dispositions.withdrawn
        missing = sorted(c for c in candidates if c not in claims)
        funnel.update(
            rejections=len(rejections),
            claims_in_dispositions=len(candidates),
            claims_missing_text=len(missing),
            negatives=len(negatives & set(claims)),
        )

        records = []
        for number in sorted(c for c in candidates if c in claims):
            rec = self._record(
                action, mail_date, number, claims, claims_source, by_claim.get(number, []),
                dispositions, aliases, pubs, cited, oard_covered, amended, listing is not None,
                statuses,
            )  # fmt: skip
            records.append(rec)
        return records, funnel

    def _with_inventor_aliases(
        self, aliases: dict[str, str], app_id: str, ifw: str
    ) -> dict[str, str]:
        """Add first-inventor names of every US document OARD lists for this application.

        Covers references the examiner names ("Shimizu in view of Kobayashi") but only
        identifies on the PTO-892 form. Names shared by two documents are ambiguous and
        dropped; names the text already defines keep the text's meaning.
        """
        rows = self.con.execute(
            "SELECT DISTINCT parsed FROM citations WHERE parsed IS NOT NULL AND "
            "(ifw_number = ? OR (app_id = ? AND form892 = '1'))", [ifw, app_id]
        ).fetchall()  # fmt: skip
        by_name: dict[str, set[str]] = {}
        for (raw,) in rows:
            norm = normalize_document_id(raw)
            if norm and norm[1] != "foreign" and norm[0].isdigit():
                if name := self.joiner.first_inventor(norm[0]):
                    by_name.setdefault(name, set()).add(norm[0])
        known = {n.lower() for n in aliases}
        extra = {name: next(iter(ids)) for name, ids in by_name.items()
                 if len(ids) == 1 and name.lower() not in known and len(name) > 2}  # fmt: skip
        return {**aliases, **extra}

    def _record(
        self,
        action: dict[str, Any],
        mail_date: str,
        number: int,
        claims: dict[int, Claim],
        claims_source: str,
        rejections: list[RejectionText],
        dispositions: Dispositions,
        aliases: dict[str, str],
        pubs: dict[str, Publication | None],
        cited: list[str],
        oard_covered: dict[int, set[str]],
        amended: bool,
        has_listing: bool,
        statuses: dict[int, str],
    ) -> ClaimRecord:
        claim = claims[number]
        parents = ancestors(number, claims)
        limitations = split_limitations(claim.text, dependent=bool(parents))
        grounds = [g for r in rejections for g in make_grounds(r, aliases)]
        flags = []
        if claims_source == "as_filed_ocr":
            flags.append("claims_as_filed_ocr")
        if not any(pub and pub.paragraphs for pub in pubs.values()):
            flags.append("no_prior_art_text")
        if any(not g.valid for g in grounds):
            unresolved = [g for g in grounds if not g.valid]
            if any(_NON_PATENT.search(g.statement) for g in unresolved):
                flags.append("ground_non_patent_literature")
            else:
                flags.append("ground_references_unresolved")
        if {d for g in grounds for d in g.references} - set(cited):
            flags.append("ground_reference_not_cited")

        if grounds:
            decision_source = "rejection_statement"
            if number in dispositions.allowable_dependent | dispositions.allowed:
                flags.append("conflicting_disposition")
        elif number in dispositions.allowed:
            decision_source = "allowed"
        else:
            decision_source = "allowable_dependent"
        decision = sorted({g.statute for g in grounds}) or ["no_rejection"]

        # Examiner text and pin cites for this claim.
        texts, evidence, overlaps, modes = [], [], [], set()
        for rej in rejections:
            windows, mode = claim_windows(rej, number)
            modes.add(mode)
            for seg in windows:
                seg_text = rej.reasoning[seg.start : seg.end].strip()
                texts.append(seg_text)
                if mode == "marker" and len(seg.claims) == 1:
                    overlaps.append(ngram_overlap(claim_body(claim.text), seg_text))
                mapped = map_evidence(limitations, rej.reasoning, seg.start, seg.end,
                                      rej.pin_cites)  # fmt: skip
                for cite, hits in mapped:
                    if cite.document_id == SELF:
                        continue
                    ev = ExaminerEvidence(cite.document_id, cite.kind, cite.raw, cite.start,
                                          cite.end, limitations=hits)  # fmt: skip
                    pub = pubs.get(cite.document_id or "")
                    if cite.kind == "paragraph" and pub and len(cite.document_id or "") == 11:
                        ev.paragraphs = resolve_paragraphs(pub, cite.start, cite.end)
                    evidence.append(ev)
        explanation_source = "examiner_reasoning" if texts else "none"
        flags += [f"reasoning_{m}" for m in sorted(modes - {"marker", "none"})]
        if not grounds:
            named = [r for r in dispositions.reasons_for_allowance if number in claims_named(r)]
            if named:
                texts, explanation_source = named, "reasons_for_allowance"
        if not texts:
            flags.append("no_reference_explanation")
        if grounds and not any(e.paragraphs for e in evidence):
            flags.append("no_paragraph_evidence")

        overlap = max((o for o in overlaps if o is not None), default=None)
        status = statuses.get(number)
        verified, verify_flags = verify_claim_text(overlap, status)
        flags += verify_flags

        covered = oard_covered.get(number)
        oard_statutes = sorted(covered) if covered is not None else None
        oard_agrees = None
        if oard_statutes is not None:
            oard_agrees = set(oard_statutes) == {g.statute for g in grounds}

        return ClaimRecord(
            example_id=f"{action['ifw_number']}:{number}",
            application_id=action["app_id"],
            office_action_id=action["ifw_number"],
            document_code=action["document_cd"],
            mail_date=mail_date,
            art_unit=action["art_unit"] or "",
            uspc_class=action.get("uspc_class") or "",
            claim_number=number,
            claim_text=claim.text,
            parent_claims=parents,
            parent_claim_texts=[claims[c].text for c in parents],
            claims_source=claims_source,
            decision_source=decision_source,
            grounds=grounds,
            decision=decision,
            art_allowable=number in dispositions.allowable_over_art,
            oard_statutes=oard_statutes,
            oard_agrees=oard_agrees,
            limitations=limitations,
            examiner_text="\n\n".join(texts),
            explanation_source=explanation_source,
            evidence=evidence,
            cited_documents=cited,
            claims_amended_before_action=amended,
            claims_listing_before_action=has_listing,
            pending_status=status,
            restatement_overlap=round(overlap, 3) if overlap is not None else None,
            claim_text_verified=verified,
            flags=flags,
        )
