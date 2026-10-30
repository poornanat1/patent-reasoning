"""Join OARD labels with office action text, published claims and prior-art full text.

For each sampled office action this produces one record per 102/103 rejection with:

- the OARD labels (statute, rejected claims) and OARD-linked citations
- the examiner's rejection statement(s) and reasoning, parsed from the office action XML
- the rejected claims' text from the application's pre-grant publication, plus a flag for
  whether the claims were amended before this action (if so, the published text may differ)
- each cited US reference's full text, and the examiner's pin cites resolved to paragraphs
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

import duckdb

from patent_reasoning.data.oard import parse_claim_numbers
from patent_reasoning.data.odp import ODPClient
from patent_reasoning.data.office_action import (
    RejectionText,
    attribute_pin_cites,
    extract_rejections,
    find_aliases,
    find_references,
    parse_blocks,
)
from patent_reasoning.data.publications import Claim, Publication, parse_publication

PRIOR_ART_STATUTES = ("102", "103")


def claims_amended_before(docs: list[dict[str, Any]], app: dict[str, Any], mail_date: str) -> bool:
    """True if the claims may differ from the as-filed (published) claims at ``mail_date``.

    The as-filed claims are a CLM document dated the filing date. Any later CLM (filed
    with an amendment or RCE) or any preliminary amendment (A.PE) before the action means
    the pending claims may not match the publication.
    """
    filed = (app.get("applicationMetaData") or {}).get("filingDate", "")[:10]
    for d in docs:
        date = d.get("officialDate", "")[:10]
        if date >= mail_date:
            continue
        code = d.get("documentCode")
        if code == "A.PE" or (code == "CLM" and filed and date > filed):
            return True
    return False


def select_office_actions(
    con: duckdb.DuckDBPyConnection, n: int, year_min: int, year_max: int, salt: str
) -> list[dict[str, Any]]:
    """Deterministically pick non-final actions with a 102/103 rejection and clean parsing flags."""
    rows = con.execute(
        f"""
        SELECT o.app_id, o.ifw_number, o.document_cd, o.mail_dt, o.art_unit, o.uspc_class
        FROM office_actions o
        WHERE o.document_cd = 'CTNF'
          AND o.rejection_fp_mismatch = '0' AND o.fp_missing = '0' AND o.header_missing = '0'
          AND year(o.mail_dt::DATE) BETWEEN {int(year_min)} AND {int(year_max)}
          AND EXISTS (SELECT 1 FROM rejections r WHERE r.ifw_number = o.ifw_number
                      AND r.action_type IN ('102', '103'))
        """
    ).fetchall()
    cols = ["app_id", "ifw_number", "document_cd", "mail_dt", "art_unit", "uspc_class"]
    actions = [dict(zip(cols, r, strict=True)) for r in rows]

    def key(a):
        return hashlib.sha256(f"{salt}:{a['ifw_number']}".encode()).hexdigest()

    return sorted(actions, key=key)[:n]


def _fulltext_uri(app: dict[str, Any], kind: str) -> str | None:
    meta = app.get("grantDocumentMetaData" if kind == "grant" else "pgpubDocumentMetaData") or {}
    return meta.get("fileLocationURI")


@dataclass
class Reference:
    document_id: str
    in_oard: bool
    in_text: bool
    alias: str | None = None
    kind: str = ""  # "patent", "publication", "foreign"
    source_app: str | None = None
    fulltext: bool = False


@dataclass
class Evidence:
    document_id: str | None
    kind: str
    start: str
    end: str
    raw: str
    offset: int = -1  # position in examiner_reasoning
    resolved: bool = False
    text: str = ""


@dataclass
class JoinedRecord:
    application_id: str
    office_action_id: str
    document_code: str
    mail_date: str
    art_unit: str
    rejection_type: str
    oard_claims: list[int]
    statements: list[str]
    examiner_reasoning: str
    claims: dict[int, str]
    claim_dependencies: dict[int, list[int]]
    claims_missing: list[int]
    claims_amended_before_action: bool
    claims_source: str = ""  # "publication" or "as_filed_ocr"
    references: list[Reference] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    status: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Joiner:
    def __init__(self, client: ODPClient, con: duckdb.DuckDBPyConnection):
        self.client = client
        self.con = con
        self._pub_cache: dict[str, tuple[str | None, Publication | None]] = {}

    # -- prior art ----------------------------------------------------------

    def _fetch_reference(self, doc_id: str) -> tuple[str | None, Publication | None]:
        """Resolve a US patent/publication number to its application and full text."""
        if doc_id in self._pub_cache:
            return self._pub_cache[doc_id]
        result: tuple[str | None, Publication | None] = (None, None)
        if len(doc_id) == 11 and doc_id.isdigit():
            queries = [f"applicationMetaData.earliestPublicationNumber:US{doc_id}{k}"
                       for k in ("A1", "A2", "A9")]  # fmt: skip
            kind = "pgpub"
        elif doc_id.isdigit():
            queries = [f"applicationMetaData.patentNumber:{doc_id}"]
            kind = "grant"
        else:
            queries, kind = [], ""
        for q in queries:
            apps = self.client.search_applications(q)
            if not apps:
                continue
            app = apps[0]
            uri = _fulltext_uri(app, kind)
            pub = None
            if uri and (xml := self.client.fulltext_xml(uri)):
                pub = parse_publication(xml)
            result = (app["applicationNumberText"], pub)
            break
        self._pub_cache[doc_id] = result
        return result

    # -- per office action --------------------------------------------------

    def join(self, action: dict[str, Any]) -> tuple[list[JoinedRecord], dict[str, Any]]:
        """Return the joined 102/103 records for one office action, plus a funnel summary."""
        app_id, ifw = action["app_id"], action["ifw_number"]
        mail_date = str(action["mail_dt"])[:10]
        funnel: dict[str, Any] = {"office_action_id": ifw, "application_id": app_id}

        docs = self.client.documents(app_id)
        oa_doc = next((d for d in docs if d.get("documentIdentifier") == ifw), None)
        funnel["oa_in_file_wrapper"] = oa_doc is not None
        xml_url = next(
            (o["downloadUrl"] for o in (oa_doc or {}).get("downloadOptionBag", [])
             if o.get("mimeTypeIdentifier") == "XML"),
            None,
        )  # fmt: skip
        funnel["oa_xml_available"] = xml_url is not None
        oa_xml = self.client.document_xml(xml_url) if xml_url else None
        funnel["oa_text"] = bool(oa_xml)
        if not oa_xml:
            return [], funnel

        blocks = parse_blocks(oa_xml)
        full_text = "\n".join(b.text for b in blocks)
        rejections = extract_rejections(blocks)
        aliases = find_aliases(full_text)
        for rej in rejections:
            attribute_pin_cites(rej, aliases)
        funnel["parsed_rejections"] = len(rejections)

        app = self.client.application(app_id) or {}
        amended = claims_amended_before(docs, app, mail_date)
        pgpub_uri = _fulltext_uri(app, "pgpub")
        own_pub = (
            parse_publication(xml)
            if pgpub_uri and (xml := self.client.fulltext_xml(pgpub_uri))
            else None
        )
        funnel["published_claims"] = bool(own_pub and own_pub.claims)
        funnel["claims_amended_before_action"] = amended
        own_claims, claims_source = (own_pub.claims, "publication") if own_pub else ({}, "")
        if not own_claims and (as_filed := self._as_filed_claims(docs, app)):
            # Unpublished applications: fall back to the OCR'd claims filed with the application.
            own_claims, claims_source = as_filed, "as_filed_ocr"
        funnel["claims_source"] = claims_source

        oard_rows = self.con.execute(
            "SELECT action_type, claim_numbers FROM rejections "
            "WHERE ifw_number = ? AND action_type IN ('102', '103')",
            [ifw],
        ).fetchall()
        oard_cites = self.con.execute(
            "SELECT DISTINCT action_type, parsed FROM citations WHERE ifw_number = ? "
            "AND citation_in_oa = '1' AND parsed IS NOT NULL",
            [ifw],
        ).fetchall()

        records = []
        for statute in PRIOR_ART_STATUTES:
            claims_oard = sorted({c for t, cn in oard_rows if t == statute
                                  for c in (parse_claim_numbers(cn) or [])})  # fmt: skip
            if not claims_oard:
                continue
            matched = [r for r in rejections if r.statute == statute]
            records.append(
                self._record(
                    action,
                    mail_date,
                    statute,
                    claims_oard,
                    matched,
                    own_claims,
                    claims_source,
                    amended,
                    {p for t, p in oard_cites if t == statute},
                    aliases,
                )  # fmt: skip
            )
        return records, funnel

    def _as_filed_claims(self, docs: list[dict[str, Any]], app: dict[str, Any]) -> dict[int, Claim]:
        filed = (app.get("applicationMetaData") or {}).get("filingDate", "")[:10]
        for d in docs:
            if d.get("documentCode") != "CLM" or d.get("officialDate", "")[:10] != filed:
                continue
            for option in d.get("downloadOptionBag", []):
                if option.get("mimeTypeIdentifier") == "XML":
                    xml = self.client.document_xml(option["downloadUrl"])
                    if xml and (claims := parse_publication(xml).claims):
                        return claims
        return {}

    def _record(
        self,
        action: dict[str, Any],
        mail_date: str,
        statute: str,
        claims_oard: list[int],
        matched: list[RejectionText],
        pub_claims: dict[int, Claim],
        claims_source: str,
        amended: bool,
        oard_doc_ids: set[str],
        aliases: dict[str, str],
    ) -> JoinedRecord:
        reasoning = "\n\n".join(r.reasoning for r in matched)
        record = JoinedRecord(
            application_id=action["app_id"],
            office_action_id=action["ifw_number"],
            document_code=action["document_cd"],
            mail_date=mail_date,
            art_unit=action["art_unit"] or "",
            rejection_type=statute,
            oard_claims=claims_oard,
            statements=[r.statement for r in matched],
            examiner_reasoning=reasoning,
            claims={c: pub_claims[c].text for c in claims_oard if c in pub_claims},
            claim_dependencies={
                c: pub_claims[c].depends_on for c in claims_oard if c in pub_claims
            },
            claims_missing=[c for c in claims_oard if c not in pub_claims],
            claims_amended_before_action=amended,
            claims_source=claims_source,
        )

        in_text = {d for _, d in find_references(reasoning)}
        alias_of = {d: a for a, d in aliases.items()}
        pubs: dict[str, Publication | None] = {}
        for doc_id in sorted(oard_doc_ids | in_text):
            kind = (
                "foreign"
                if not doc_id.isdigit()
                else ("publication" if len(doc_id) == 11 else "patent")
            )
            ref = Reference(
                doc_id, doc_id in oard_doc_ids, doc_id in in_text, alias_of.get(doc_id), kind
            )
            if kind != "foreign":
                ref.source_app, pubs[doc_id] = self._fetch_reference(doc_id)
                ref.fulltext = pubs[doc_id] is not None
            record.references.append(ref)

        base = 0  # offset of each rejection's text within the joined reasoning
        for rej in matched:
            for cite in rej.pin_cites:
                ev = Evidence(
                    cite.document_id, cite.kind, cite.start, cite.end, cite.raw, base + cite.offset
                )
                pub = pubs.get(cite.document_id or "")
                # Only publications print paragraph numbers; granted patents are cited by
                # column and line, so a "paragraph N" cite to a grant can't be trusted.
                is_publication = len(cite.document_id or "") == 11
                if cite.kind == "paragraph" and pub is not None and is_publication:
                    nums = [n for n in pub.paragraphs if cite.start <= n <= cite.end]
                    if nums:
                        ev.resolved = True
                        ev.text = " ".join(pub.paragraphs[n] for n in nums[:5])
                record.evidence.append(ev)
            base += len(rej.reasoning) + 2

        record.status = {
            "statement_found": bool(matched),
            "claims_found": bool(record.claims) and not record.claims_missing,
            "claims_reliable": bool(record.claims) and not record.claims_missing and not amended,
            "prior_art_text": any(r.fulltext for r in record.references),
            "paragraph_evidence": any(e.resolved for e in record.evidence),
        }
        record.status["traceable"] = (
            record.status["statement_found"]
            and record.status["claims_found"]
            and record.status["prior_art_text"]
        )
        record.status["evidence_grounded"] = (
            record.status["traceable"]
            and record.status["claims_reliable"]
            and record.status["paragraph_evidence"]
        )
        return record
