from patent_reasoning.data.sft import build_example

DOC = "20130173955"
DOCUMENTS = {
    DOC: {"document_id": DOC, "kind": "us_publication", "title": "Disk arrays", "passages": [
        {"passage_id": f"{DOC}:0092", "locator": "[0092]", "text": "A disk may be unavailable."},
        {"passage_id": f"{DOC}:0131", "locator": "[0131]", "text": "Aborting a rebuild process."},
    ]},
    "4690653": {"document_id": "4690653", "kind": "us_patent", "title": "", "passages": []},
}  # fmt: skip
ALL = [f"{DOC}:0092", f"{DOC}:0131"]


def _record(**overrides):
    record = {
        "example_id": "OA1:2",
        "application_id": "13624988",
        "office_action_id": "OA1",
        "document_code": "CTNF",
        "mail_date": "2014-10-22",
        "art_unit": "2137",
        "uspc_class": "711",
        "claim_number": 2,
        "claim_text": "2. The method of claim 1, further comprising delaying a rebuild process.",
        "parent_claims": [1],
        "parent_claim_texts": ["1. A method comprising writing data to a spare disk."],
        "claims_source": "publication",
        "decision_source": "rejection_statement",
        "grounds": [{"statute": "102", "references": [DOC], "statement": "...", "valid": True}],
        "decision": ["102"],
        "oard_agrees": True,
        "limitations": ["delaying a rebuild process"],
        "examiner_text": "Regarding claim 2, Hallak discloses delaying a rebuild (para. 131).",
        "explanation_source": "examiner_reasoning",
        "evidence": [{"document_id": DOC, "kind": "paragraph", "raw": "para. 131",
                      "start": "0131", "end": "0131", "paragraphs": ["0131"],
                      "limitations": [0]}],
        "cited_documents": [DOC],
        "claims_amended_before_action": False,
        "claim_text_verified": True,
        "claim_text_reliable": True,
        "flags": [],
    }  # fmt: skip
    record.update(overrides)
    return record


def build(record, passages=ALL):
    return build_example(record, DOCUMENTS, passages, "action_cited", max_target_tokens=700)


def test_rejection_with_visible_evidence_is_supported():
    out = build(_record())
    target = out.example.target
    assert target.decision == {"102"} and target.evidence_status == "supported"
    (m,) = target.claim_mapping
    assert (m.disclosed, m.document_id, m.passage_ids) == (True, DOC, [f"{DOC}:0131"])
    assert out.example.provenance.pin_cites[0].resolved_passage_ids == [f"{DOC}:0131"]


def test_mapping_never_cites_passages_outside_the_input():
    out = build(_record(limitations=["delaying a rebuild process", "logging the delay"],
                        evidence=[{**_record()["evidence"][0], "paragraphs": ["0131", "0092"],
                                   "limitations": [0, 1]}]), passages=[f"{DOC}:0131"])  # fmt: skip
    assert all(p == f"{DOC}:0131" for m in out.example.target.claim_mapping for p in m.passage_ids)


def test_rejection_without_visible_evidence_is_excluded():
    column_line = [{"document_id": "4690653", "kind": "column_line", "raw": "col. 3, l. 5-9",
                    "start": "3:5", "end": "3:9", "paragraphs": [],
                    "limitations": [0]}]  # fmt: skip
    out = build(_record(grounds=[{"statute": "102", "references": ["4690653"], "statement": "",
                                  "valid": True}], evidence=column_line,
                        cited_documents=[DOC, "4690653"]))  # fmt: skip
    assert out.example is None and out.reason == "evidence_not_in_input"
    # cited passages that didn't fit the budget
    assert build(_record(), passages=[f"{DOC}:0092"]).reason == "evidence_not_in_input"


def test_single_limitation_allowable_claim_maps_as_not_disclosed():
    out = build(_record(decision_source="allowable_dependent", grounds=[], evidence=[],
                        decision=["no_rejection"], examiner_text="",
                        explanation_source="none"))  # fmt: skip
    target = out.example.target
    assert target.decision == {"no_rejection"} and target.evidence_status == "not_applicable"
    assert [(m.disclosed, m.document_id) for m in target.claim_mapping] == [(False, None)]
    assert target.explanation is None


def test_allowable_claim_needs_prior_art_text():
    out = build(_record(decision_source="allowed", grounds=[], evidence=[]), passages=[])
    assert out.reason == "no_prior_art_text"


def test_112_rejection_needs_no_prior_art():
    out = build(_record(grounds=[{"statute": "112", "references": [], "statement": "",
                                  "valid": True}], evidence=[]), passages=[])  # fmt: skip
    assert out.example.target.decision == {"112"}
    assert out.example.target.claim_mapping == []


def test_unresolved_labels_are_excluded_with_a_reason():
    assert build(_record(claim_text_reliable=False)).reason == "claim_text_unreliable"
    assert build(_record(flags=["ground_references_unresolved"])).reason == "references_unresolved"
    assert build(_record(flags=["ground_non_patent_literature"])).reason == "non_patent_literature"


def test_duplicate_grounds_collapse():
    g = {"statute": "103", "references": [DOC], "statement": "", "valid": True}
    out = build(_record(grounds=[g, dict(g)]))
    assert len(out.example.target.grounds) == 1


def test_explanation_about_another_claim_is_excluded():
    off = "Shaknovich and Weber teach the limitations of claim 4, upon which claim 5 depends."
    assert build(_record(examiner_text=off)).reason == "explanation_off_claim"
