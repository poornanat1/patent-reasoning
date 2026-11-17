import json
from pathlib import Path

from jsonschema import Draft202012Validator

from patent_reasoning.model.output import generation_schema
from patent_reasoning.schema import Example, LimitationMapping
from patent_reasoning.train.sft_records import records, target


def _example(name: str) -> Example:
    return Example.model_validate_json(Path(f"docs/schemas/examples/{name}").read_text())


def _records(examples, max_tokens=10_000):
    return records(examples, render=lambda m: m[-1]["content"], count_tokens=len,
                   max_tokens=max_tokens, eos="<|im_end|>")  # fmt: skip


def test_a_fully_mapped_rejection_becomes_a_record_that_fits_the_grammar():
    e = _example("rejection_102.json")
    out, excluded = _records([e])
    assert not excluded and out[0]["completion"].endswith("<|im_end|>")
    data = json.loads(out[0]["completion"].removesuffix("<|im_end|>"))
    assert Draft202012Validator(generation_schema(e.input.limitations)).is_valid(data)


def test_art_rejections_without_a_full_mapping_are_excluded():
    e = _example("rejection_102.json")
    empty = e.model_copy(update={"target": e.target.model_copy(update={"claim_mapping": []})})
    assert target(empty) == (None, "art_mapping_empty")
    extra = e.input.model_copy(update={"limitations": [*e.input.limitations, "a valve"]})
    assert target(e.model_copy(update={"input": extra})) == (None, "art_mapping_partial")


def test_an_allowance_without_reasons_keeps_an_empty_explanation():
    e = _example("rejection_102.json")
    allowed = e.target.model_copy(update={"grounds": [], "explanation": None,
                                          "evidence_status": "not_applicable",
                                          "claim_mapping": [LimitationMapping(
                                              limitation=lim, disclosed=False)
                                              for lim in e.input.limitations]})  # fmt: skip
    t, reason = target(e.model_copy(update={"target": allowed}))
    assert reason is None and t.explanation == ""
    rejected = e.target.model_copy(update={"explanation": None})
    assert target(e.model_copy(update={"target": rejected})) == (
        None, "rejection_without_explanation")  # fmt: skip


def test_examples_over_the_token_budget_are_excluded_not_truncated():
    out, excluded = _records([_example("rejection_102.json")], max_tokens=10)
    assert out == [] and excluded == {"over_max_tokens": 1}
