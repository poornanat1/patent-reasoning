from patent_reasoning.data.publications import parse_publication

PUB_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE us-patent-application SYSTEM "us-patent-application-v42-2006-08-23.dtd" [ ]>
<us-patent-application>
<us-bibliographic-data-application>
<publication-reference><document-id><country>US</country><doc-number>20080149277</doc-number>
<kind>A1</kind></document-id></publication-reference>
<invention-title>Manufacturing apparatus</invention-title>
</us-bibliographic-data-application>
<abstract><p id="p-0001" num="0000">An apparatus.</p></abstract>
<description><heading>BACKGROUND</heading>
<p id="p-0002" num="0001">The present invention relates to <i>transdermal</i> preparations.</p>
<p id="p-0003" num="0002">A runway carries the sheet.</p></description>
<claims>
<claim id="CLM-00001" num="00001"><claim-text>1. An apparatus comprising:
<claim-text>a runway.</claim-text></claim-text></claim>
<claim id="CLM-00002" num="00002"><claim-text>2. The apparatus of
<claim-ref idref="CLM-00001">claim 1</claim-ref>, wherein the runway is long.</claim-text></claim>
</claims>
</us-patent-application>"""


def test_parse_publication():
    pub = parse_publication(PUB_XML)
    assert (pub.doc_number, pub.kind, pub.title) == ("20080149277", "A1", "Manufacturing apparatus")
    assert pub.paragraphs == {
        "0001": "The present invention relates to transdermal preparations.",
        "0002": "A runway carries the sheet.",
    }
    assert pub.claims[1].text == "1. An apparatus comprising: a runway."
    assert pub.claims[1].independent
    assert pub.claims[2].depends_on == [1]
    assert pub.abstract == "An apparatus."
