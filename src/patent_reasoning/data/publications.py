"""Parse USPTO full-text XML (pre-grant publications and grants, "Red Book" v4.x).

Gives claims with dependencies and description paragraphs keyed by their printed
paragraph number, which is what examiners cite as "[0023]" for publications.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import etree

_PARSER = etree.XMLParser(
    recover=True, resolve_entities=False, load_dtd=False, no_network=True, huge_tree=True
)


def _text(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


@dataclass
class Claim:
    number: int
    text: str
    depends_on: list[int] = field(default_factory=list)

    @property
    def independent(self) -> bool:
        return not self.depends_on


@dataclass
class Publication:
    doc_number: str  # e.g. "20080149277" or "08327903"
    kind: str  # A1, B1, B2, ...
    title: str
    claims: dict[int, Claim]
    paragraphs: dict[str, str]  # printed number ("0023") -> text
    abstract: str


def parse_publication(xml: str) -> Publication:
    root = etree.fromstring(xml.encode("utf-8"), _PARSER)
    pub_ref = root.find(".//publication-reference/document-id")
    doc_number = pub_ref.findtext("doc-number", "") if pub_ref is not None else ""
    kind = pub_ref.findtext("kind", "") if pub_ref is not None else ""

    claims: dict[int, Claim] = {}
    for el in root.iterfind(".//claims/claim"):
        try:
            num = int(el.get("num", ""))
        except ValueError:
            continue
        refs = [r.get("idref", "") for r in el.iter("claim-ref")]
        deps = sorted({int(m.group(1)) for r in refs if (m := re.search(r"CLM-0*(\d+)", r))})
        claims[num] = Claim(num, _text(el), deps)

    paragraphs = {}
    for el in root.iterfind(".//description//p"):
        if num := el.get("num"):
            paragraphs[num] = _text(el)

    abstract_el = root.find(".//abstract")
    title_el = root.find(".//invention-title")
    return Publication(
        doc_number=doc_number,
        kind=kind,
        title=_text(title_el) if title_el is not None else "",
        claims=claims,
        paragraphs=paragraphs,
        abstract=_text(abstract_el) if abstract_el is not None else "",
    )
