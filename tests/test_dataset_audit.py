import json
from pathlib import Path

from patent_reasoning.data.dataset_audit import (
    check,
    conflicting_duplicates,
    disposition_sentences,
    sample,
    stratum,
)
from patent_reasoning.data.office_action import Block
from patent_reasoning.schema import Example


def _example(name: str = "rejection_102.json", **updates) -> Example:
    data = json.loads(Path(f"docs/schemas/examples/{name}").read_text())
    for path, value in updates.items():
        *keys, last = path.split(".")
        node = data
        for key in keys:
            node = node[key]
        node[last] = value
    return Example.model_validate(data)


def test_clean_example_passes_checks():
    assert check(_example()) == []
    assert stratum(_example()) == "102"


def test_check_finds_missing_parent_and_off_claim_limitation():
    e = _example(**{"input.parent_claims": []})
    assert "parent_claim_missing" in check(e)
    mapping = _example().target.claim_mapping[0].model_dump()
    mapping["limitation"] = "a limitation from some other claim"
    e = _example(**{"target.claim_mapping": [mapping]})
    assert "limitation_not_in_claim" in check(e)


def test_a_claim_naming_itself_is_not_a_missing_parent():
    claim = {"number": 2, "text": "2. The method of claim 2, further comprising inhibiting or "
             "delaying a rebuild process on disks in said given enclosure."}  # fmt: skip
    assert "parent_claim_missing" not in check(_example(**{"input.claim": claim}))


def test_conflicting_duplicates_need_the_same_input_and_another_decision():
    a = _example()
    b = a.model_copy(update={"example_id": "X:2"})
    assert conflicting_duplicates([a, b]) == set()
    allowed = a.target.model_copy(update={"grounds": [], "claim_mapping": [],
                                          "evidence_status": "not_applicable"})  # fmt: skip
    c = a.model_copy(update={"example_id": "Y:2", "target": allowed})
    assert conflicting_duplicates([a, b, c]) == {a.example_id, "X:2", "Y:2"}


def test_sample_takes_every_decision_class_before_repeating_one():
    rejected = [_example().model_copy(update={"example_id": f"R:{i}"}) for i in range(5)]
    allowed = _example("allowable_dependent.json")
    picked = sample({"train": [*rejected, allowed]}, {"train": 2}, "s")
    assert {stratum(e) for e in picked} == {"102", "no_rejection"}
    assert sample({"train": rejected}, {"train": 3}, "s") == sample(
        {"train": list(reversed(rejected))}, {"train": 3}, "s"
    )


def test_disposition_sentences_find_claims_in_ranges_and_statements_without_a_verb():
    blocks = [
        Block(None, "Claims 1-15 are rejected under 35 U.S.C. 101. The art is discussed below."),
        Block(None, "Claims 3, 8 and 13 rejected under 35 U.S.C. 103(a) as being unpatentable."),
        Block(None, "Claim 20 is objected to as being dependent upon a rejected base claim."),
    ]
    found = disposition_sentences(blocks, 8)
    assert len(found) == 2 and "101" in found[0] and "103" in found[1]
    assert disposition_sentences(blocks, 20)[0].startswith("Claim 20 is objected")
