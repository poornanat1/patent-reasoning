from patent_reasoning.data.joined import claims_amended_before


def _doc(code, date):
    return {"documentCode": code, "officialDate": f"{date}T00:00:00.000-0500"}


def test_claims_amended_before():
    app = {"applicationMetaData": {"filingDate": "2008-06-11"}}
    as_filed = [_doc("CLM", "2008-06-11"), _doc("SPEC", "2008-06-11")]
    assert not claims_amended_before(as_filed, app, "2009-12-29")
    assert claims_amended_before(as_filed + [_doc("CLM", "2010-06-28")], app, "2011-02-03")
    # an amendment after the action doesn't count
    assert not claims_amended_before(as_filed + [_doc("CLM", "2010-06-28")], app, "2009-12-29")
    assert claims_amended_before(as_filed + [_doc("A.PE", "2008-06-11")], app, "2009-12-29")
