import json
from pathlib import Path

from patent_reasoning.eval.decision import bootstrap, metrics, paired_bootstrap, predicted
from patent_reasoning.eval.faithfulness import check, score
from patent_reasoning.schema import Example

R102, R103, NONE = frozenset({"102"}), frozenset({"103"}), frozenset({"no_rejection"})


def _example() -> Example:
    return Example.model_validate_json(Path("docs/schemas/examples/rejection_102.json").read_text())


def _output(passage_ids, document_id="20130173955", status="supported", disclosed=True):
    example = _example()
    return json.dumps({
        "claim_mapping": [{"limitation": example.input.limitations[0], "disclosed": disclosed,
                           "document_id": document_id, "passage_ids": passage_ids}],
        "explanation": "x",
        "grounds": [{"statute": "102", "references": ["20130173955"]}],
        "evidence_status": status,
    })  # fmt: skip


def test_predicted_reads_grounds_or_no_rejection_and_none_for_garbage():
    assert predicted('{"grounds": [{"statute": "103"}, {"statute": "112"}]}') == {"103", "112"}
    assert predicted('{"grounds": []}') == NONE
    assert predicted('{"grounds": [') is None


def test_metrics_score_each_class_and_the_error_rates():
    gold = [R102, R103 | {"112"}, NONE, NONE]
    pred = [R102, R103, R103, None]
    m = metrics(gold, pred)
    assert m["per_class"]["102"]["f1"] == 1.0
    assert m["per_class"]["112"]["recall"] == 0.0
    assert m["per_class"]["103"]["precision"] == 0.5
    assert m["false_rejection_rate"] == 0.5  # one of two allowable claims rejected
    assert m["false_allowance_rate"] == 0.0
    assert m["unreadable_rate"] == 0.25
    assert m["exact_match"] == 0.25


def test_bootstrap_resamples_applications_and_brackets_the_estimate():
    gold = [R102, R103, NONE, R102] * 10
    pred = [R102, R102, NONE, R102] * 10
    apps = [f"A{i // 2}" for i in range(40)]
    low, high = bootstrap(gold, pred, apps, n=200)
    assert low <= metrics(gold, pred)["macro_f1"] <= high
    same = paired_bootstrap(gold, pred, pred, apps, n=50)
    assert same["difference"] == 0 and same["ci95"] == [0, 0]


def test_existing_supporting_citation_counts_as_valid():
    c = check(_example(), _output(["20130173955:0131"]))
    s = score(c)
    assert s["citation_existence"] == 1.0
    assert s["unsupported_assertion_rate"] == 0.0
    assert s["support_overclaim_rate"] == 0.0
    assert s["examiner_pin_recall"] == 1.0


def test_missing_or_misattributed_passages_are_invalid():
    s = score(check(_example(), _output(["20130173955:9999", "4690653:0001"])))
    assert s["citation_existence"] == 0.0
    assert s["unsupported_assertion_rate"] == 1.0
    assert s["art_rejection_without_citation_rate"] == 1.0
    assert s["support_overclaim_rate"] == 1.0
    assert s["examiner_pin_recall"] == 0.0


def test_unreadable_output_contributes_nothing():
    assert check(_example(), "{") == {}
