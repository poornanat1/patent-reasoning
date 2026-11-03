import json
from pathlib import Path

from patent_reasoning.data.families import families
from patent_reasoning.data.splits import assign, family_split, fully_grounded
from patent_reasoning.schema import Example


def test_families_join_through_shared_links():
    links = {
        "A": {"app:A", "app:PCTEP2007"},  # national stage of a PCT application
        "B": {"app:B", "app:PCTEP2007"},
        "C": {"app:C", "priority:JP:2004123"},
        "D": {"app:D", "priority:JP:2004123", "app:C"},
        "E": {"app:E"},
    }
    fam = families(links)
    assert fam["A"] == fam["B"] and fam["C"] == fam["D"]
    assert len({fam["A"], fam["C"], fam["E"]}) == 3
    # stable: the ID depends only on the members
    assert families(dict(reversed(list(links.items())))) == fam


def test_family_split_is_deterministic_and_roughly_proportional():
    splits = [family_split(f"fam-{i}", "s", 0.1, 0.1) for i in range(2000)]
    assert splits == [family_split(f"fam-{i}", "s", 0.1, 0.1) for i in range(2000)]
    assert 0.07 < splits.count("val") / 2000 < 0.13
    assert 0.07 < splits.count("test") / 2000 < 0.13


def _example(name, **provenance):
    data = json.loads(Path(f"docs/schemas/examples/{name}").read_text())
    data["provenance"].update(provenance)
    return Example.model_validate(data)


def test_eval_splits_keep_only_fully_grounded_examples():
    good = _example("rejection_102.json")
    shaky = _example("allowable_dependent.json", flags=["mapping_partial"])
    assert fully_grounded(good) and not fully_grounded(shaky)
    family_of = {"13624988": "fam-x"}
    # force the family into an evaluation split by holding out its technology center
    _, split_of_example, dropped = assign([good, shaky], family_of, "s", 0.1, 0.1, ["21"])
    assert split_of_example == {good.example_id: "ood_test"}
    assert dropped == [shaky.example_id]


def test_ood_holdout_takes_the_whole_family():
    a = _example("rejection_102.json")
    b = _example("allowable_dependent.json", application_id="99999999", art_unit="2800")
    family_of = {"13624988": "fam-x", "99999999": "fam-x"}
    split_of_family, split_of_example, _ = assign([a, b], family_of, "s", 0.1, 0.1, ["21"])
    assert split_of_family == {"fam-x": "ood_test"}
    assert set(split_of_example.values()) == {"ood_test"}
