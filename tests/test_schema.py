import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from patent_reasoning.schema import (
    NO_REJECTION,
    Example,
    Ground,
    LimitationMapping,
    ModelOutput,
)

EXAMPLES = Path("docs/schemas/examples")


def load(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text())


@pytest.mark.parametrize(
    "name, decision",
    [("rejection_102.json", {"102"}), ("allowable_dependent.json", {NO_REJECTION})],
)
def test_real_examples_validate(name, decision):
    ex = Example.model_validate(load(name))
    assert ex.target.decision == decision


def test_exported_schemas_are_current():
    sys.path.insert(0, "scripts")
    from export_schemas import OUT, SCHEMAS, render

    for name, model in SCHEMAS.items():
        assert (OUT / name).read_text() == render(model), f"run scripts/export_schemas.py ({name})"


@pytest.mark.parametrize(
    "statute, refs",
    [("102", []), ("102", ["a", "b"]), ("103", []), ("112", ["a"]), ("101", ["a"])],
)
def test_ground_reference_counts(statute, refs):
    with pytest.raises(ValidationError):
        Ground(statute=statute, references=refs)


def test_disclosed_limitation_needs_a_document():
    with pytest.raises(ValidationError):
        LimitationMapping(limitation="a runway", disclosed=True)
    assert not LimitationMapping(limitation="a runway", disclosed=False).document_id


@pytest.mark.parametrize(
    "grounds, status, ok",
    [
        ([], "not_applicable", True),
        ([], "supported", False),
        ([{"statute": "112"}], "not_applicable", True),
        ([{"statute": "103", "references": ["d"]}], "not_applicable", False),
        (
            [{"statute": "103", "references": ["d"]}, {"statute": "112"}],
            "partially_supported",
            True,
        ),
    ],
)
def test_evidence_status_matches_decision(grounds, status, ok):
    data = {"grounds": grounds, "evidence_status": status}
    if ok:
        assert ModelOutput.model_validate(data)
    else:
        with pytest.raises(ValidationError):
            ModelOutput.model_validate(data)


def test_multi_ground_decision():
    out = ModelOutput.model_validate(
        {"grounds": [{"statute": "103", "references": ["d"]}, {"statute": "112"}],
         "evidence_status": "supported"}
    )  # fmt: skip
    assert out.decision == {"103", "112"}


def _mutate(name: str, fn) -> dict:
    data = load(name)
    fn(data)
    return data


@pytest.mark.parametrize(
    "name, mutation",
    [
        # target cites a passage the model never saw
        ("rejection_102.json",
         lambda d: d["target"]["claim_mapping"][0]["passage_ids"].append("20130173955:0200")),
        # ground relies on a document not in the input
        ("rejection_102.json", lambda d: d["target"]["grounds"][0].update(references=["7654321"])),
        # example_id must be office action id + claim number
        ("rejection_102.json", lambda d: d.update(example_id="I1HUBKC5PXXIFW4:3")),
        # an allowable claim can't carry rejection grounds
        ("allowable_dependent.json", lambda d: d["target"].update(
            grounds=[{"statute": "102", "references": ["20130173955"]}],
            evidence_status="supported")),
        # unknown fields are rejected, so typos don't silently drop data
        ("allowable_dependent.json", lambda d: d["input"].update(office_action="leak")),
    ],
)  # fmt: skip
def test_example_consistency_checks(name, mutation):
    with pytest.raises(ValidationError):
        Example.model_validate(_mutate(name, mutation))
