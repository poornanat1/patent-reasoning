import json
from pathlib import Path

from patent_reasoning.data.preferences import grounded_in_input, make_pairs
from patent_reasoning.schema import Example


def _example(name):
    return Example.model_validate(json.loads(Path(f"docs/schemas/examples/{name}").read_text()))


REJECTED = _example("rejection_102.json")  # claim 2, 102 over Hallak, mapped to [0131]
ALLOWABLE = _example("allowable_dependent.json")


def _pairs(example, limitations):
    return {p.type: p for p in make_pairs(example, "train", limitations)}


def test_rejection_pairs():
    lims = [m.limitation for m in REJECTED.target.claim_mapping]
    pairs = _pairs(REJECTED, lims)
    assert set(pairs) == {"citation_corruption", "unsupported_rejection", "false_allowance"}
    corrupt = pairs["citation_corruption"].rejected
    assert corrupt.claim_mapping[0].passage_ids == ["20130173955:0092"]  # visible, not cited
    assert corrupt.decision == REJECTED.target.decision
    assert all(not m.passage_ids for m in pairs["unsupported_rejection"].rejected.claim_mapping)
    assert pairs["false_allowance"].rejected.decision == {"no_rejection"}


def test_false_rejection_of_an_allowable_claim():
    lims = ["the rebuild process is suspended while the enclosure is unavailable"]
    pair = _pairs(ALLOWABLE, lims)["false_rejection"]
    assert pair.example.target.decision == {"no_rejection"}
    assert pair.rejected.decision == {"102"}
    assert pair.rejected.evidence_status == "supported"


def test_overconfident_support_invents_the_missing_mappings():
    partial = REJECTED.target.model_copy(update={"evidence_status": "partially_supported"})
    example = REJECTED.model_copy(update={"target": partial})
    lims = [REJECTED.target.claim_mapping[0].limitation, "logging each delayed rebuild"]
    rejected = _pairs(example, lims)["overconfident_support"].rejected
    assert rejected.evidence_status == "supported"
    assert [m.limitation for m in rejected.claim_mapping] == lims


def test_every_rejected_output_stays_inside_the_input():
    for example, lims in [(REJECTED, ["inhibiting a rebuild"]), (ALLOWABLE, ["a spare disk"])]:
        for pair in make_pairs(example, "train", lims):
            assert grounded_in_input(example, pair.rejected)
            assert pair.rejected != example.target
