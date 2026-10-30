import pytest

from patent_reasoning.data import oard


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1-3, 5", [1, 2, 3, 5]),
        ("7", [7]),
        ("2;1", [1, 2]),
        ("claims 1 and 4", None),
        ("5-3", None),
        (None, None),
    ],
)
def test_parse_claim_numbers(value, expected):
    assert oard.parse_claim_numbers(value) == expected


def test_in_sample_is_deterministic_and_close_to_rate():
    ids = [f"{i:08d}" for i in range(20_000)]
    picked = [a for a in ids if oard.in_sample(a, 0.1)]
    assert picked == [a for a in ids if oard.in_sample(a, 0.1)]
    assert 0.09 < len(picked) / len(ids) < 0.11


def test_find_download_uri_walks_nested_response():
    product = {"bulkDataProductBag": [{"productFileBag": {"fileDataBag": [
        {"fileName": "rejections.csv.zip", "fileDownloadURI": "https://x/rej"},
    ]}}]}  # fmt: skip
    assert oard.find_download_uri(product, "rejections.csv.zip") == "https://x/rej"
    with pytest.raises(KeyError, match="rejections.csv.zip"):
        oard.find_download_uri(product, "citations.csv.zip")


def test_download_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("USPTO_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="USPTO_API_KEY"):
        oard.download_tables(tmp_path)


def test_ids_keep_leading_zeros(csvs):
    con = oard.connect(csvs)
    assert con.execute("SELECT min(app_id) FROM office_actions").fetchone()[0] == "01234567"


def test_build_sample_respects_cohort_and_keeps_all_tables(csvs, tmp_path):
    counts = oard.build_sample(oard.connect(csvs), tmp_path / "sample", 1.0, 2008, 2014)
    assert counts == {
        "cohort_apps": 2,
        "sampled_apps": 2,
        "office_actions": 3,
        "rejections": 3,
        "citations": 3,
    }
