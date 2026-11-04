import pytest

from patent_reasoning.data.office_action import (
    attribute_pin_cites,
    extract_rejections,
    find_aliases,
    find_pin_cites,
    find_references,
    parse_blocks,
    parse_claim_list,
)

# Trimmed from a real CTFR (application 12/000,742), same element structure.
OA_XML = """<?xml version="1.0" encoding="UTF-8"?>
<uspat:OutgoingDocument xmlns:uspat="urn:us:gov:doc:uspto:patent"
  xmlns:com="http://www.wipo.int/standards/XMLSchema/ST96/Common"
  xmlns:uscom="urn:us:gov:doc:uspto:common">
<uspat:DocumentMetadata><uscom:DocumentCode>CTFR</uscom:DocumentCode></uspat:DocumentMetadata>
<uscom:FormParagraph><uscom:FormParagraphNumber>USC_103_heading</uscom:FormParagraphNumber>
<uscom:P><com:B>Claim Rejections - 35 USC &#167; 103</com:B></uscom:P></uscom:FormParagraph>
<uscom:FormParagraph><uscom:FormParagraphNumber>07-21</uscom:FormParagraphNumber>
<uscom:P>Claims 1, 5 and 10-12 are rejected under 35 U.S.C. 103(a) as being unpatentable over
US Patent 5,688,523 (Garbe hereinafter) in view of US 20040126415 (Lu hereinafter).
Regarding claim 1, Garbe discloses a sheet (column 2, lines 63-66). Lu teaches a drug
(paragraph [0096]).<?PageStart number='2'?></uscom:P></uscom:FormParagraph>
<uscom:P>Garbe further discloses a runway (col. 8, ll. 25-31).</uscom:P>
<uscom:FormParagraph><uscom:FormParagraphNumber>07-22</uscom:FormParagraphNumber>
<uscom:P>Claim 9 is rejected under 35 U.S.C. 103(a) as being unpatentable over Garbe in view
of EP 1449527 (Miranda hereinafter). Miranda discloses a second sheet (paragraph 8).</uscom:P>
</uscom:FormParagraph>
<uscom:FormParagraph><uscom:FormParagraphNumber>conclusion_heading</uscom:FormParagraphNumber>
<uscom:P>Conclusion</uscom:P></uscom:FormParagraph>
<uscom:P>Any inquiry concerning this communication should be directed to the examiner.</uscom:P>
<uscom:BoundaryDataBag><uscom:P>Application/Control Number: 12/000,742 Page 2</uscom:P>
</uscom:BoundaryDataBag>
</uspat:OutgoingDocument>"""


@pytest.fixture
def rejections():
    blocks = parse_blocks(OA_XML)
    rejs = extract_rejections(blocks)
    aliases = find_aliases("\n".join(b.text for b in blocks))
    for r in rejs:
        attribute_pin_cites(r, aliases)
    return rejs


def test_blocks_drop_metadata_numbers_and_page_markers():
    blocks = parse_blocks(OA_XML)
    assert [b.fp_number for b in blocks[:2]] == ["USC_103_heading", "07-21"]
    assert all("Control Number" not in b.text for b in blocks)
    assert not blocks[1].text.startswith("07-21")
    assert "number=" not in blocks[1].text


def test_extracts_statements_claims_and_trailing_reasoning(rejections):
    assert [(r.statute, r.claims) for r in rejections] == [
        ("103", [1, 5, 10, 11, 12]),
        ("103", [9]),
    ]
    assert rejections[0].statement.endswith("(Lu hereinafter).")
    assert "runway" in rejections[0].reasoning  # untagged paragraph after the statement
    assert "inquiry" not in rejections[1].reasoning  # stops at the heading


def test_pin_cites_are_attributed_to_the_right_reference(rejections):
    got = [(c.kind, c.start, c.end, c.document_id) for r in rejections for c in r.pin_cites]
    assert got == [
        ("column_line", "2:63", "2:66", "5688523"),
        ("paragraph", "0096", "0096", "20040126415"),
        ("column_line", "8:25", "8:31", "5688523"),
        ("paragraph", "0008", "0008", "EP1449527"),
    ]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Smith (US Patent #7679495)", ["7679495"]),
        ("U.S. Pat. No. 7,706,944", ["7706944"]),
        ("US PG Pub. #20040204807", ["20040204807"]),
        ("U.S. 2004/0126415 A1", ["20040126415"]),
        ("WO 2005/012345 and JP 2004-123456", ["WO2005012345", "JP2004123456"]),
        ("claims 1-3, 5 and 7", []),
        ("Schubert ( 6, 817, 711 B2) in view of Simone (5, 720, 040)", ["6817711", "5720040"]),
        ("anticipated by Spahrbier 3843410.", ["3843410"]),
        ("over US patent 6127444 by Kadri", ["6127444"]),
        ("Parnell (Des. 373,240) in view of Willows et al. (D 497,245)", ["D373240", "D497245"]),
        ("about 1000000 cells per well", []),
        ("anticipated by US patent 5,003,185, hereafter Burgio.", ["5003185"]),
        ("over EP 1 192 959 as applied to claims 1, 3-4 above", ["EP1192959"]),
        ("by European Patent Application 1 455 128 A1 to Scanferia", ["EP1455128"]),
    ],
)
def test_find_references(text, expected):
    assert [d for _, d in find_references(text)] == expected


def test_parse_claim_list():
    assert parse_claim_list("1-5, 7 and 9 through 11") == [1, 2, 3, 4, 5, 7, 9, 10, 11]


def test_pin_cite_forms():
    cites = find_pin_cites("(col. 3, lines 5-9) ([0012]-[0014]) (Col 4 lines 1-5) (¶ 21)")
    assert [(c.kind, c.start, c.end) for c in cites] == [
        ("column_line", "3:5", "3:9"),
        ("paragraph", "0012", "0014"),
        ("column_line", "4:1", "4:5"),
        ("paragraph", "0021", "0021"),
    ]
    assert find_pin_cites("Diamond v. Diehr, 450 U.S. 175 (1981); see (0092).")[0].start == "0092"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Claims 5-7, 9-12 & 20-24 are rejected under 35 U.S.C. 103(a) as obvious.",
         ("103", [5, 6, 7, 9, 10, 11, 12, 20, 21, 22, 23, 24])),
        ("Claim(s) 1-3 is/are rejected under 35 U.S.C. 102(b) as anticipated.", ("102", [1, 2, 3])),
        ("Claims 3, 8 and 13 rejected under pre-AIA 35 U.S.C. 103(a) as obvious.",
         ("103", [3, 8, 13])),
        ("Claims 1-3 and 10 , per above incorporated art rejections, are in the alternative, are "
         "also rejected under pre-AIA 35 U.S.C. 102(e) as being taught by Gillenson.",
         ("102", [1, 2, 3, 10])),
    ],
)  # fmt: skip
def test_statement_variants(text, expected):
    from patent_reasoning.data.office_action import Block

    (rej,) = extract_rejections([Block("07-21", text)])
    assert (rej.statute, rej.claims) == expected


def test_aliases_with_name_suffixes():
    text = (
        "over Wong et al. (US 2007/0266003) in view of Aslanian, JR. (US 2004/0254859) "
        "and Grim, III et al. (US 2006/0085254 A1, 04/20/2006)."
    )
    assert find_aliases(text) == {
        "Wong": "20070266003",
        "Aslanian": "20040254859",
        "Grim": "20060085254",
    }


def test_aliases_tolerate_reference_lists():
    # regression: every digit in the window is followed by a digit or comma
    text = "Kubacki, Ramirez, GB 2480710, GB 2406786, Bodling teach bed-tent shelters."
    assert find_aliases(text) == {"Ramirez": "GB2480710"}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("over US 2010/0283633 A1 to Becker in view of Laubinger", {"Becker": "20100283633"}),
        ("over Rouse US Patent Application No. US 2012/0159182 A1", {"Rouse": "20120159182"}),
        ("over Yaegashi et al. (USPG 2005/0258463)", {"Yaegashi": "20050258463"}),
        ("over Koyama et al (U.S. PG Pub. 2007/0019864)", {"Koyama": "20070019864"}),
        ("over Fini et al. (USP No. 2011/0279888).", {"Fini": "20110279888"}),
    ],
)
def test_alias_forms(text, expected):
    assert find_aliases(text) == expected


def test_foreign_numbers_do_not_leak_us_numbers():
    text = (
        "over (EP 1,621,137 A2) and GU (KR10-2004-0042965) in view of Voglewede (US 2005/0268638)"
    )
    assert [d for _, d in find_references(text)] == ["EP1621137", "KR1020040042965", "20050268638"]


def test_more_pin_cite_forms():
    cites = find_pin_cites("(see 7:11-35) (par. 11) teaches (0045) but a ratio of 1:2")
    assert [(c.kind, c.start, c.end) for c in cites] == [
        ("column_line", "7:11", "7:35"),
        ("paragraph", "0011", "0011"),
        ("paragraph", "0045", "0045"),
    ]


def test_cites_to_own_specification_are_tagged_self():
    from patent_reasoning.data.office_action import SELF, RejectionText

    text = "Koyama (US 2007/0019864) teaches a reader (¶12); see also (Spec., ¶30) for the unit."
    rej = RejectionText("103", [1], "", text, find_pin_cites(text))
    attribute_pin_cites(rej, find_aliases(text))
    assert [c.document_id for c in rej.pin_cites] == ["20070019864", SELF]


def test_statement_is_not_cut_at_initials():
    from patent_reasoning.data.office_action import Block

    text = ("Claim 7 is rejected under 35 U.S.C. 103(a) as being unpatentable over Scheiner in "
            "view of St. Germain and F.C. Smith. Scheiner discloses a lid.")  # fmt: skip
    (rej,) = extract_rejections([Block("07-21", text)])
    assert rej.statement.endswith("F.C. Smith.")
