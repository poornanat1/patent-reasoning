import pytest

from patent_reasoning.data import oard
from patent_reasoning.data.audit import render_markdown, run_audit


def test_audit_catches_known_issues(csvs, tmp_path):
    result = run_audit(oard.connect(csvs), full=True)
    assert all(not s["missing_columns"] for s in result["schema"])
    ids = result["identifiers"]
    assert ids["rej_orphan_ifw"] == 1
    assert ids["oa_duplicate_ifw_numbers"] == 0
    # Linked citations with a blank action_subtype must still match their rejection row.
    assert ids["cit_linked_orphan_actions"] == 0
    assert result["dates"]["unparsed_rate"] == 0
    assert result["labels"]["claim_numbers_parse_rate"] == pytest.approx(4 / 5)
    agree = {r["flag"]: r for r in result["labels"]["flag_vs_rejection_rows"]}
    assert agree["rejection_103"]["both"] == 2

    sample_dir = tmp_path / "sample"
    oard.build_sample(oard.connect(csvs), sample_dir, 1.0, 2008, 2014)
    sample = {t: sample_dir / f"{t}.parquet" for t in oard.TABLES}
    md = render_markdown(run_audit(oard.connect(sample), full=False), "test")
    assert "## Citations" in md
