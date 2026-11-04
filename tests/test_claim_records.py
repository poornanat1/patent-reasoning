import pytest

from patent_reasoning.data.claim_records import (
    ancestors,
    claim_windows,
    make_grounds,
    map_evidence,
    ngram_overlap,
    parse_dispositions,
    split_limitations,
    verify_claim_text,
)
from patent_reasoning.data.office_action import Block, RejectionText, find_pin_cites
from patent_reasoning.data.publications import Claim, parse_claim_statuses


def test_dispositions():
    d = parse_dispositions([
        Block("12-151-08", "Claims 4-9 and 16 are objected to as being dependent upon a rejected "
              "base claim, but would be allowable if rewritten in independent form including all "
              "of the limitations of the base claim and any intervening claims."),
        Block("12-151-07", "Claims 21-22 are allowed."),
        Block("07-43-02", "Claim 13 would be allowable if rewritten to overcome the rejection(s) "
              "under 35 U.S.C. 112, 2nd paragraph, set forth in this Office action."),
        Block("08-06", "Claims 30-31 are withdrawn from further consideration pursuant to 37 CFR "
              "1.142(b), there being no allowable generic or linking claim."),
        Block("08-06", "Claims 40-41 and 44 have been withdrawn from further consideration."),
        Block("07-05-05", "Applicant is advised that should claims 1-9 be found allowable, claims "
              "10-20 will be objected to under 37 CFR 1.75."),
        Block("13-03-01", "The following is a statement of reasons for the indication of "
              "allowable subject matter: Claims 5 and 14 have been indicated allowable because "
              "the prior art fails to teach a hinge."),
        Block("conclusion_heading", "Conclusion"),
    ])  # fmt: skip
    assert d.allowable_dependent == {4, 5, 6, 7, 8, 9, 16}
    assert d.allowed == {21, 22}
    assert d.allowable_over_art == {13}
    assert d.withdrawn == {30, 31, 40, 41, 44}
    assert len(d.reasons_for_allowance) == 1 and "hinge" in d.reasons_for_allowance[0]


def _rej(statute, claims, statement, reasoning=""):
    text = statement + (" " + reasoning if reasoning else "")
    return RejectionText(statute, claims, statement, text, find_pin_cites(text))


def _grounds(statute, statement):
    aliases = {"Smith": "5688523", "Jones": "20040126415"}
    return [(g.statute, g.references, g.valid)
            for g in make_grounds(_rej(statute, [1], statement), aliases)]  # fmt: skip


def test_grounds():
    rejected = "Claim 1 is rejected under 35 U.S.C."
    assert _grounds(
        "103", f"{rejected} 103(a) as being unpatentable over Smith in view of Jones."
    ) == [("103", ["5688523", "20040126415"], True)]
    # no identifiable reference
    assert _grounds("102", f"{rejected} 102(b) as anticipated by Routh (cited by Applicant).") == [
        ("102", [], False)
    ]
    assert _grounds("102", f"{rejected} 102(b) as being anticipated by or, in the alternative, "
                    "under 35 U.S.C. 103(a) as obvious over Smith.") == [
        ("102", ["5688523"], True), ("103", ["5688523"], True)
    ]  # fmt: skip
    assert _grounds(
        "112", f"{rejected} 112, second paragraph, as indefinite over Smith's usage."
    ) == [("112", [], True)]
    # an explicit number beats an alias defined earlier for another document
    smith = f"{rejected} 102(b) as anticipated by Smith (US Patent 7,343,588, hereinafter Smith)."
    assert _grounds("102", smith) == [("102", ["7343588"], True)]
    # a priority application is not a reference
    angele = (f"{rejected} 102(b) as anticipated by Angele, US 2006/0167856 A1, which claims "
              "foreign priority to application EP 05001407.5.")  # fmt: skip
    assert _grounds("102", angele) == [("102", ["20060167856"], True)]


def test_split_limitations():
    independent = (
        "1. A method of monitoring a transaction between a client and a server, comprising: "
        "upon receipt of a page from the server, executing a script to generate a value; and "
        "returning the value to a proxy, wherein the proxy logs the value."
    )
    assert split_limitations(independent, dependent=False) == [
        "upon receipt of a page from the server, executing a script to generate a value",
        "returning the value to a proxy",
        "wherein the proxy logs the value",
    ]
    dependent = ("2. The method according to claim 1 further comprising inhibiting or delaying a "
                 "rebuild process on disks in said given enclosure.")  # fmt: skip
    assert split_limitations(dependent, dependent=True) == [
        "inhibiting or delaying a rebuild process on disks in said given enclosure"
    ]
    assert split_limitations("3. The method of claim 2, wherein the disks are SSDs.", True) == [
        "the disks are SSDs"
    ]
    for lim in split_limitations(independent, dependent=False):
        assert lim in independent  # verbatim


def test_ancestors_fall_back_to_text_for_ocr_claims():
    claims = {
        1: Claim(1, "1. A device comprising a hinge."),
        2: Claim(2, "2 (original). The device of claim 1, wherein the hinge is steel."),
        3: Claim(3, "3. The device as claimed in claim 2, further comprising a latch."),
    }
    assert ancestors(3, claims) == [2, 1]
    assert ancestors(1, claims) == []


STATEMENT = "Claims 1-5 are rejected under 35 U.S.C. 102(b) as being anticipated by Smith."


def test_claim_windows():
    rej = _rej("102", [1, 2, 3, 4, 5], STATEMENT,
               "Smith discloses a device comprising a hinge (col. 2, lines 5-9) and a latch "
               "mounted on the frame of the device (col. 3, lines 1-4). Regarding claim 2, Smith "
               "discloses the hinge is steel (col. 4, lines 1-2). The latch is spring loaded "
               "[as in claims 3-4].")  # fmt: skip
    (lead,), mode = claim_windows(rej, 1)  # discussed before any marker
    assert mode == "marker" and "latch mounted" in rej.reasoning[lead.start : lead.end]
    (seg,), mode = claim_windows(rej, 2)
    assert mode == "marker" and rej.reasoning[seg.start :].startswith("Regarding claim 2")
    (seg,), mode = claim_windows(rej, 4)
    assert mode == "mention" and "spring loaded" in rej.reasoning[seg.start : seg.end]
    assert claim_windows(rej, 5) == ([], "none")
    shared = _rej("112", [6, 7], "Claims 6-7 are rejected under 35 U.S.C. 112, second paragraph.",
                  "The term 'substantially' renders the claims indefinite because no standard "
                  "for measuring the degree is given.")  # fmt: skip
    assert claim_windows(shared, 7)[1] == "shared"


def test_map_evidence():
    text = ("Smith discloses a device comprising a steel hinge (col. 2, lines 5-9) and a latch "
            "mounted on the frame (col. 3, lines 1-4).")  # fmt: skip
    limitations = ["a steel hinge", "a latch mounted on the frame", "a battery"]
    mapped = map_evidence(limitations, text, 0, len(text), find_pin_cites(text))
    assert [(c.start, hits) for c, hits in mapped] == [("2:5", [0]), ("3:1", [1])]


def test_map_evidence_stops_at_figure_cites():
    text = ("Smith discloses a steel hinge (Fig. 13) and a latch mounted on the frame "
            "(paragraph [0071]).")  # fmt: skip
    mapped = map_evidence(["a steel hinge", "a latch mounted on the frame"], text, 0, len(text),
                          find_pin_cites(text))  # fmt: skip
    assert [(c.start, hits) for c, hits in mapped] == [("0071", [1])]


def test_restatement_overlap_ignores_interleaved_cites():
    claim = "a hinge connecting the lid to the base and a latch holding the lid closed"
    examiner = ("Smith teaches a hinge (12) connecting the lid to the base [0031] and a latch "
                "(14) holding the lid closed (fig. 2).")  # fmt: skip
    assert ngram_overlap(claim, examiner) == 1.0
    assert ngram_overlap(claim, "Smith teaches all limitations; see the rejection of claim 1.") == 0


@pytest.mark.parametrize(
    "overlap, status, expected",
    [
        (None, None, None),
        (0.9, None, True),
        (0.1, None, None),  # examiners paraphrase; low overlap alone proves nothing
        (0.9, "currently amended", False),  # the amendment's status label decides
        (None, "original", True),
    ],
)
def test_verify_claim_text(overlap, status, expected):
    assert verify_claim_text(overlap, status)[0] is expected


def test_parse_claim_statuses():
    xml = """<doc><claims>
      <claim num="1"><claim-text>1. (Currently Amended) A device comprising a hinge.</claim-text></claim>
      <claim num="2"><claim-text>2 (original). The device of claim 1.</claim-text></claim>
      <claim><claim-text><boundary-data type="line-number">5</boundary-data>3. (Cancelled)</claim-text></claim>
      <claim><claim-text>4.(New) The device of claim 1.</claim-text></claim>
    </claims></doc>"""  # noqa: E501
    assert parse_claim_statuses(xml) == {
        1: "currently amended", 2: "original", 3: "canceled", 4: "new"
    }  # fmt: skip


def test_bracketed_claim_headings_start_a_segment():
    rej = _rej("103", [4, 5], "Claims 4-5 are rejected under 35 U.S.C. 103(a) over Smith.",
               "[Claim 4] Smith discloses a balloon with a hydrophilic surface (para. 29). "
               "[Claim 5] Smith teaches a duration of 30 to 90 seconds (para. 41).")  # fmt: skip
    (seg,), mode = claim_windows(rej, 5)
    assert mode == "marker"
    assert rej.reasoning[seg.start : seg.end].startswith("[Claim 5]")
    assert "hydrophilic" not in rej.reasoning[seg.start : seg.end]
