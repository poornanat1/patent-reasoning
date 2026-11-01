"""Prior-art evidence: normalized document IDs, passages, and the fixed retrieval budget.

Every experiment sees the same passages for a claim, chosen under a fixed token budget from
the documents cited in the office action. Two conditions:

- ``action_cited`` (primary): every passage the examiner pin-cited anywhere in the action,
  then BM25 fill. The cited pool is the same for every claim in the action, rejected or
  allowable, so it doesn't reveal any claim's decision; it shows the examiner's passages,
  not the examiner's conclusions.
- ``retrieved``: BM25 only, queried with the claim's limitations and the claim and its
  parents as a whole. The office action is not used at all.

Ranking never looks at the office action or the label.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from patent_reasoning.data.publications import Publication

FOREIGN_CODES = ("EP", "WO", "JP", "DE", "GB", "FR", "CN", "KR", "CA", "AU")

# ---------------------------------------------------------------------------
# Document IDs
# ---------------------------------------------------------------------------


def normalize_document_id(raw: str | None) -> tuple[str, str] | None:
    """(document_id, kind) in the form the office action parser produces, or None.

    Accepts OARD's ``parsed`` values ("20060025037", "4690653", "WO 03068528") and loose
    forms such as "US 2004/0126415 A1", "US 6,817,711 B2", "D 497,245" or "RE41,233".
    Kinds match ``schema.DocumentKind``.
    """
    if not raw:
        return None
    s = re.sub(r"[\s.]", "", raw.upper())
    s = re.sub(r"(?<=\d)[A-Z]\d?$", "", s)  # kind code: A1, B2, E, ...
    if m := re.fullmatch(rf"({'|'.join(FOREIGN_CODES)})([\d/,\-]+)", s):
        return m.group(1) + re.sub(r"\D", "", m.group(2)), "foreign"
    s = s.removeprefix("US")
    if m := re.fullmatch(r"(DES|D|RE|PP)0*(\d[\d,]*)", s):
        return {"DES": "D"}.get(m.group(1), m.group(1)) + m.group(2).replace(",", ""), "us_patent"
    digits = re.sub(r"[,/\-]", "", s)
    if not digits.isdigit():
        return None
    if len(digits) == 11 and digits[:2] in ("19", "20"):
        return digits, "us_publication"
    if len(digits) <= 8 and int(digits):
        return str(int(digits)), "us_patent"
    return None


def document_kind(doc_id: str) -> str:
    if doc_id[:2] in FOREIGN_CODES:
        return "foreign"
    if len(doc_id) == 11 and doc_id.isdigit():
        return "us_publication"
    return "us_patent"


# ---------------------------------------------------------------------------
# Passages
# ---------------------------------------------------------------------------


def approx_tokens(text: str) -> int:
    """Rough token count (4 characters per token), used for budgeting before the base
    model's tokenizer is fixed. Patent prose runs close to this ratio."""
    return max(1, math.ceil(len(text) / 4))


def truncate(text: str, max_tokens: int) -> str:
    limit = max_tokens * 4
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + " …"


@dataclass
class PassageRecord:
    passage_id: str
    locator: str
    text: str
    tokens: int


def document_passages(
    doc_id: str, pub: Publication | None, max_passage_tokens: int
) -> list[PassageRecord]:
    """A document's abstract and description paragraphs, in reading order.

    Publications print their paragraph numbers, so the locator is how examiners cite them
    ("[0092]"). Grants are cited by column and line, which the XML doesn't record; their
    paragraphs get a locator by position that no examiner citation will match.
    """
    if pub is None:
        return []
    out = []
    if pub.abstract:
        text = truncate(pub.abstract, max_passage_tokens)
        out.append(PassageRecord(f"{doc_id}:abstract", "Abstract", text, approx_tokens(text)))
    printed = document_kind(doc_id) == "us_publication"
    for num, para in pub.paragraphs.items():
        if not para.strip():
            continue
        text = truncate(para, max_passage_tokens)
        position = int(num) if num.isdigit() else num
        locator = f"[{num}]" if printed else f"description paragraph {position}"
        out.append(PassageRecord(f"{doc_id}:{num}", locator, text, approx_tokens(text)))
    return out


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

_STOPWORDS = set(
    "the a an of to and or in on for with by at from as is are be said that which wherein "
    "each one least further comprising including having claim claims thereof such being "
    "into than then this these those its it not can may when where".split()
)


def terms(text: str) -> list[str]:
    words = re.findall(r"[a-z][a-z0-9\-]+", text.lower())
    return [w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w
            for w in words if w not in _STOPWORDS]  # fmt: skip


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avg = sum(self.lengths) / len(docs) if docs else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        q = Counter(query)
        out = []
        for tf, length in zip(self.tf, self.lengths, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg or 1))
            out.append(sum(
                w * self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
                for t, w in q.items() if t in tf
            ))  # fmt: skip
        return out


@dataclass(frozen=True)
class Budget:
    """The evidence budget shared by every experiment."""

    max_tokens: int = 2000
    max_passages: int = 12
    max_passage_tokens: int = 300
    min_per_document: int = 1
    claim_weight: int = 2  # the claim's own terms count this many times its parents'
    limitation_depth: int = 2  # top passages taken per limitation before general fill
    cited_share: float = 0.7  # most of the budget the examiner-cited pool may take


def claim_query(claim_text: str, parent_texts: list[str], weight: int) -> list[str]:
    return terms(claim_text) * weight + [t for p in parent_texts for t in terms(p)]


@dataclass
class Selection:
    passage_ids: list[str] = field(default_factory=list)
    tokens: int = 0


def select_passages(
    query: list[str],
    documents: dict[str, list[PassageRecord]],
    budget: Budget,
    limitation_queries: list[list[str]] = (),
    cited: frozenset[str] | set[str] = frozenset(),
) -> Selection:
    """Pick passages for one claim within ``budget``.

    In order, until the budget is spent:

    1. passages in ``cited`` (the action's examiner-cited pool), best match to the claim
       first, up to ``cited_share`` of the budget
    2. each document's ``min_per_document`` best passages, so no cited reference is
       invisible
    3. the ``limitation_depth`` best passages for each limitation, round robin
    4. the best passages for the claim as a whole

    The result is in document order, then reading order.
    """
    pool = [(doc_id, i, p) for doc_id, ps in documents.items() for i, p in enumerate(ps)]
    if not pool:
        return Selection()
    bm25 = BM25([terms(p.text) for _, _, p in pool])
    scores = bm25.scores(query)
    ranked = sorted(range(len(pool)), key=lambda k: (-scores[k], k))

    chosen: list[int] = []
    tokens = 0

    def take(k: int, limit: int) -> None:
        nonlocal tokens
        if k in chosen or len(chosen) >= budget.max_passages:
            return
        if tokens + pool[k][2].tokens > limit:
            return
        chosen.append(k)
        tokens += pool[k][2].tokens

    for k in ranked:
        if pool[k][2].passage_id in cited:
            take(k, int(budget.max_tokens * budget.cited_share))
    for doc_id in documents:
        for k in [k for k in ranked if pool[k][0] == doc_id][: budget.min_per_document]:
            take(k, budget.max_tokens)
    per_limitation = []
    for q in limitation_queries:
        s = bm25.scores(q)
        per_limitation.append(sorted(range(len(pool)), key=lambda k: (-s[k], k)))
    for depth in range(budget.limitation_depth):
        for ranking in per_limitation:
            if depth < len(ranking):
                take(ranking[depth], budget.max_tokens)
    for k in ranked:
        take(k, budget.max_tokens)
    order = {d: n for n, d in enumerate(documents)}
    chosen.sort(key=lambda k: (order[pool[k][0]], pool[k][1]))
    return Selection([pool[k][2].passage_id for k in chosen], tokens)
