"""Parse office action XML from the file wrapper (``xmlarchive`` downloads).

An office action is a sequence of blocks. Form paragraphs carry a number (e.g. ``07-21``
is a 103 rejection statement); untagged paragraphs are free examiner text. From the
blocks we extract each rejection's statute, claims and reasoning, the prior-art
documents it names, and the pin cites (column/line or paragraph) the examiner relies on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import etree

_PARSER = etree.XMLParser(recover=True, resolve_entities=False, no_network=True, huge_tree=True)
_SKIP = {"DocumentMetadata", "BoundaryDataBag", "FormParagraphNumber"}


@dataclass
class Block:
    fp_number: str | None
    text: str


@dataclass
class PinCite:
    kind: str  # "paragraph" or "column_line"
    start: str  # paragraph number ("0023") or "col:line"
    end: str
    offset: int  # position in the rejection's reasoning text
    raw: str
    document_id: str | None = None


@dataclass
class RejectionText:
    statute: str  # "101", "102", "103", "112"
    claims: list[int]
    statement: str
    reasoning: str
    pin_cites: list[PinCite] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def _localname(el) -> str:
    return etree.QName(el).localname if isinstance(el.tag, str) else ""


# Embedded figures leave placeholders like "PNG media_image1.png 269 586 media_image1.png
# Greyscale" in the text.
_IMAGE_PLACEHOLDER = re.compile(
    r"\b(?:PNG|JPE?G|TIFF?|GIF)\s+(media_image\d+\.\w+)\s+\d+\s+\d+\s+\1\s+\w+"
)


def _block_text(el) -> str:
    """Text of a block without its form paragraph number. Mutates ``el``."""
    for node in el.iter():
        if _localname(node) in _SKIP:
            node.text = ""
    text = re.sub(r"\s+", " ", "".join(el.itertext(tag=etree.Element)))
    return re.sub(r"\s+", " ", _IMAGE_PLACEHOLDER.sub(" ", text)).strip()


def parse_blocks(xml: str) -> list[Block]:
    root = etree.fromstring(xml.encode("utf-8"), _PARSER)
    blocks = []
    for child in root:
        name = _localname(child)
        if name in _SKIP or not name:
            continue
        fp = None
        for node in child.iter():
            if _localname(node) == "FormParagraphNumber":
                fp = (node.text or "").strip()
                break
        text = _block_text(child)
        if text or fp:
            blocks.append(Block(fp, text))
    return blocks


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------

_STATEMENT = re.compile(
    r"Claims?(?:\(s\))?\s+(?P<claims>\d[\d,;\s\-–&and/through]*?)\s+(?:is/are|is|are|stand|stands)\s+"
    r"(?:also\s+|further\s+|again\s+)?rejected\s+under\s+(?:pre-AIA\s+)?35\s*U\.?\s*S\.?\s*C\.?"
    r"\s*§*\s*(?P<statute>101|102|103|112)",
    re.IGNORECASE,
)


def parse_claim_list(text: str) -> list[int]:
    """Expand "1-5, 7 and 9 through 11" to [1, 2, 3, 4, 5, 7, 9, 10, 11]."""
    claims: set[int] = set()
    norm = re.sub(r"\s*(?:-|–|through)\s*", "-", text)
    for lo, hi in re.findall(r"(\d+)(?:-(\d+))?", norm):
        start, end = int(lo), int(hi or lo)
        if start <= end <= start + 500:
            claims.update(range(start, end + 1))
    return sorted(claims)


_ABBREVIATIONS = {
    "u", "s", "u.s", "u.s.c", "pat", "no", "nos", "app", "appl", "pub", "publ", "al", "et",
    "fig", "figs", "col", "cols", "para", "paras", "e.g", "i.e", "ser", "seq", "vs",
}  # fmt: skip


def sentence_end(text: str, start: int) -> int:
    """Index just past the sentence that contains ``start``, skipping citation abbreviations."""
    for m in re.finditer(r"\.(?=\s+[A-Z(]|\s*$)", text[start:]):
        word = re.search(r"([A-Za-z.]+)$", text[: start + m.start()])
        if word and word.group(1).lower().rstrip(".") in _ABBREVIATIONS:
            continue
        return start + m.end()
    return len(text)


def is_heading(block: Block) -> bool:
    return bool(block.fp_number) and (
        block.fp_number.endswith("_heading") or block.fp_number in {"OacsClose"}
    )


def extract_rejections(blocks: list[Block]) -> list[RejectionText]:
    """Each rejection statement plus the untagged examiner text that follows it."""
    out: list[RejectionText] = []
    current: RejectionText | None = None
    for block in blocks:
        m = _STATEMENT.search(block.text)
        if m:
            current = RejectionText(
                statute=m.group("statute"),
                claims=parse_claim_list(m.group("claims")),
                statement=block.text[m.start() : sentence_end(block.text, m.end())],
                reasoning=block.text,
            )
            out.append(current)
        elif is_heading(block):
            current = None
        elif current is not None and block.text:
            current.reasoning += "\n" + block.text
    for rej in out:
        rej.pin_cites = find_pin_cites(rej.reasoning)
    return out


# ---------------------------------------------------------------------------
# Prior-art references and pin cites
# ---------------------------------------------------------------------------

# Patent "5,688,523" anywhere, or a bare 7-8 digit number after a cue such as "US", "Pat.",
# "No." or "#" ("US Patent #7679495", "U.S. Pat. No. 7679495"); publication "2004/0126415".
_PATENT = re.compile(
    r"(?<![\d/,])(\d{1,2},\d{3},\d{3})(?![\d,])"
    r"|(?:U\.?\s?S\.?|USPN|Pat(?:ent)?\.?|No\.?|#)[\s#.:]*(?:Pat(?:ent)?\.?\s*)?(?:No\.?\s*)?#?\s*"
    r"(\d{7,8})(?![\d/])"
)
_PUB = re.compile(r"(?<!\d)((?:19|20)\d{2})\s?[/-]?\s?(\d{7})(?!\d)")

# Foreign documents (EP 1449527, EP 1,621,137, WO 2005/012345, KR10-2004-0042965) can't be
# fetched from the USPTO, but must be recognized so their pin cites aren't attributed to a
# US reference and their digits aren't read as US numbers.
_FOREIGN = re.compile(r"\b(EP|WO|JP|DE|GB|FR|CN|KR|CA|AU)\s?(\d[\d/,\-]{4,16}\d)(?![\d/])")


def normalize_patent(number: str) -> str:
    return str(int(re.sub(r"\D", "", number)))


def find_references(text: str) -> list[tuple[int, str]]:
    """(offset, normalized document number) for every patent/publication number mentioned.

    US patents normalize to their number without commas ("5688523"), US publications to
    11 digits ("20040126415"), foreign documents to country code + digits ("EP1449527").
    """
    foreign = list(_FOREIGN.finditer(text))
    refs = [(m.start(), m.group(1) + re.sub(r"\D", "", m.group(2))) for m in foreign]

    def inside_foreign(pos: int) -> bool:
        return any(m.start() <= pos < m.end() for m in foreign)

    refs += [
        (m.start(), m.group(1) + m.group(2))
        for m in _PUB.finditer(text)
        if not inside_foreign(m.start())
    ]
    taken = {o for o, _ in refs}
    for m in _PATENT.finditer(text):
        number = m.group(1) or m.group(2)
        start = m.start(1) if m.group(1) else m.start(2)
        if not inside_foreign(start) and not any(abs(start - o) < 12 for o in taken):
            refs.append((start, normalize_patent(number)))
    return sorted(refs)


_NAME = r"(?P<name>[A-Z][A-Za-z'’\-]+)"
_ALIAS_AFTER = re.compile(
    r"^[^().;]{0,60}?\(\s*(?:hereinafter|hereafter|herein after|herein|referred to as)?"
    rf",?\s*[\"“']?{_NAME}",
)
_ALIAS_HEREINAFTER = re.compile(rf"^[^.;]{{0,60}}?(?:hereinafter|hereafter)\s*[\"“']?{_NAME}")
# "2010/0283633 A1 to Becker", "6055537 by Jack J. LeTourneau" (takes the first capitalized word)
_ALIAS_TO = re.compile(rf"^\s*(?:[AB]\d)?\s*,?\s*(?:to|by|issued to)\s+{_NAME}")
# "Garbe (US 5,688,523)", "Rouse US Patent Application No. US 2012/...", "USPGPub ... to X"
_DOC_WORDS = (
    r"US|U\.\s?S|USP|USPN|USPG|PG|PGPub|USPGPub|Pat|Patent|Pub|Publication|Application|App|"
    r"Appl|Pre-?Grant|Published|No|Number|Nos|Ser|Serial"
)
_ALIAS_BEFORE = re.compile(
    rf"{_NAME}(?:,?\s+(?:Jr|JR|Sr|SR|II|III|IV)\.?)?(?:\s+et\.?\s+al\.?)?,?\s*[(\[]?"
    rf"(?:\s*(?:{_DOC_WORDS})\.?)*[\s#:.,]*$"
)
_NOT_NAMES = {
    "US", "U", "S", "USP", "USPN", "USPG", "PG", "PGPub", "USPGPub", "Patent", "Pat",
    "Publication", "Pub", "Published", "No", "Application", "App", "Claim", "Claims", "In",
    "View", "And", "Over", "The", "See", "Of", "Number", "Jr", "JR", "II", "III", "IV",
    "EP", "WO", "JP", "DE", "GB", "FR", "CN", "KR", "CA", "AU", "European", "Japanese",
}  # fmt: skip


def find_aliases(text: str) -> dict[str, str]:
    """Map examiner shorthand (e.g. "Garbe") to the document numbers it introduces."""
    aliases: dict[str, str] = {}
    for offset, doc_id in find_references(text):
        after = text[offset : offset + 120]
        # skip past the number itself (last digit not followed by more of the number)
        if number_end := re.search(r"\d(?![\d,/\-])", after):
            after = after[number_end.end() :]
        before = text[max(0, offset - 80) : offset]
        for m in (
            _ALIAS_HEREINAFTER.match(after),
            _ALIAS_AFTER.match(after),
            _ALIAS_TO.match(after),
            _ALIAS_BEFORE.search(before),
        ):
            if m and m.group("name") not in _NOT_NAMES:
                aliases.setdefault(m.group("name"), doc_id)
                break
    return aliases


_COL_LINE = re.compile(
    r"(?:col(?:umn)?s?\.?|c\.)\s*(?P<col>\d{1,3})\s*[,:]?\s*(?:lines?|ll?\.|ln\.?|l\.)\s*"
    r"(?P<l1>\d{1,3})(?:\s*(?:-|–|to|through)\s*(?:(?:col(?:umn)?\.?\s*(?P<col2>\d{1,3})\s*,?\s*"
    r"(?:lines?|ll?\.)?\s*)?(?P<l2>\d{1,3})))?",
    re.IGNORECASE,
)
# Short form "7:11-35" (column 7, lines 11-35). Requires a line range so ratios like "1:2"
# aren't read as citations.
_COL_LINE_SHORT = re.compile(
    r"(?<![\d:])(?P<col>\d{1,2}):(?P<l1>\d{1,2})\s*[-–]\s*(?P<l2>\d{1,2})\b"
)
_PARA = re.compile(
    r"(?:(?:paragraphs?|paras?\.?|pars?\.|¶+)\s*\[?(?P<p1>\d{1,4})\]?|\[(?P<b1>\d{4})\]"
    r"|\((?P<b2>\d{4})\))"
    r"(?:\s*(?:-|–|to|through)\s*\[?(?P<p2>\d{1,4})\]?)?",
    re.IGNORECASE,
)


def find_pin_cites(text: str) -> list[PinCite]:
    cites = []
    for m in _COL_LINE.finditer(text):
        col2 = m.group("col2") or m.group("col")
        end = f"{col2}:{m.group('l2')}" if m.group("l2") else f"{m.group('col')}:{m.group('l1')}"
        cites.append(
            PinCite("column_line", f"{m.group('col')}:{m.group('l1')}", end, m.start(), m.group(0))
        )
    long_spans = [(c.offset, c.offset + len(c.raw)) for c in cites]
    for m in _COL_LINE_SHORT.finditer(text):
        if not any(a <= m.start() < b for a, b in long_spans):
            col = m.group("col")
            cites.append(PinCite("column_line", f"{col}:{m.group('l1')}",
                                 f"{col}:{m.group('l2')}", m.start(), m.group(0)))  # fmt: skip
    for m in _PARA.finditer(text):
        p1 = (m.group("p1") or m.group("b1") or m.group("b2")).zfill(4)
        p2 = (m.group("p2") or p1).zfill(4)
        cites.append(PinCite("paragraph", p1, p2, m.start(), m.group(0)))
    return sorted(cites, key=lambda c: c.offset)


# Cites to the application under examination, e.g. "(Spec., ¶30)" or "instant specification".
SELF = "SELF"
_SELF_CUE = re.compile(
    r"(?:\bSpec(?:ification)?\.?|instant (?:application|disclosure)|applicant'?s'? "
    r"(?:disclosure|specification)|as filed)[\s,:;]*(?:page \d+,?\s*)?$",
    re.IGNORECASE,
)


def attribute_pin_cites(rej: RejectionText, aliases: dict[str, str]) -> None:
    """Assign each pin cite to the closest preceding mention of a prior-art document.

    Cites introduced by "Spec." or similar refer to the application itself and get ``SELF``.
    """
    mentions = find_references(rej.reasoning)
    for name, doc_id in aliases.items():
        mentions += [
            (m.start(), doc_id) for m in re.finditer(rf"\b{re.escape(name)}\b", rej.reasoning)
        ]
    mentions.sort()
    for cite in rej.pin_cites:
        if _SELF_CUE.search(rej.reasoning[max(0, cite.offset - 40) : cite.offset]):
            cite.document_id = SELF
            continue
        preceding = [d for o, d in mentions if o < cite.offset]
        if preceding:
            cite.document_id = preceding[-1]
