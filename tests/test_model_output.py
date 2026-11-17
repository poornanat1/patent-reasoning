import json
from pathlib import Path

from jsonschema import Draft202012Validator

from patent_reasoning.model.output import FIELD_ORDER, generation_schema, parse, serialize
from patent_reasoning.model.prompt import messages, render_input
from patent_reasoning.schema import Example, Ground


def _example(name: str = "rejection_102.json") -> Example:
    return Example.model_validate_json(Path(f"docs/schemas/examples/{name}").read_text())


def test_serialize_follows_field_order_and_round_trips():
    target = _example().target
    text = serialize(target)
    assert list(json.loads(text)) == list(FIELD_ORDER)
    assert parse(text).output == target


def _valid(schema: dict, data: dict) -> bool:
    return Draft202012Validator(schema).is_valid(data)


def _out(mapping=(), grounds=(), status="not_applicable") -> dict:
    return {"claim_mapping": list(mapping), "explanation": "x", "grounds": list(grounds),
            "evidence_status": status}  # fmt: skip


def _entry(limitation: str) -> dict:
    return {"limitation": limitation, "disclosed": True, "document_id": "D1",
            "passage_ids": ["D1:0001"]}  # fmt: skip


R102 = {"statute": "102", "references": ["D1"]}
R103 = {"statute": "103", "references": ["D1", "D2"]}
R112 = {"statute": "112", "references": []}


def test_schema_mapping_is_empty_or_one_entry_per_limitation_in_order():
    schema = generation_schema(["a sensor", "a valve"])
    assert _valid(schema, _out())
    assert _valid(schema, _out([_entry("a sensor"), _entry("a valve")]))
    assert not _valid(schema, _out([_entry("a sensor")]))
    assert not _valid(schema, _out([_entry("a valve"), _entry("a sensor")]))
    assert not _valid(schema, _out([_entry("a sensor")] * 3))
    assert not _valid(generation_schema([]), _out([_entry("a sensor")]))


def test_schema_requires_every_field_in_order():
    for variant in generation_schema(["a sensor"])["anyOf"]:
        assert list(variant["properties"]) == variant["required"] == list(FIELD_ORDER)
        assert variant["properties"]["explanation"]["type"] == "string"


def test_schema_fixes_each_statutes_reference_count():
    schema = generation_schema(["a sensor"])
    assert _valid(schema, _out(grounds=[R102], status="supported"))
    assert not _valid(schema, _out(grounds=[R102 | {"references": ["D1", "D2"]}],
                                   status="supported"))  # fmt: skip
    assert not _valid(schema, _out(grounds=[R103 | {"references": []}], status="supported"))
    assert not _valid(schema, _out(grounds=[R112 | {"references": ["D1"]}]))


def test_schema_ties_the_evidence_status_to_the_grounds():
    schema = generation_schema(["a sensor"])
    assert _valid(schema, _out(grounds=[R112]))
    assert _valid(schema, _out(grounds=[R103, R112], status="unsupported"))
    assert not _valid(schema, _out(status="unsupported"))  # no rejection with a support level
    assert not _valid(schema, _out(grounds=[R112], status="supported"))
    assert not _valid(schema, _out(grounds=[R103]))  # art rejection marked not applicable
    assert not _valid(schema, _out(grounds=[R112, R103], status="supported"))  # art goes first


def test_only_an_allowance_may_have_an_empty_explanation():
    schema = generation_schema(["a sensor"])
    assert _valid(schema, _out() | {"explanation": ""})
    assert not _valid(schema, _out(grounds=[R112]) | {"explanation": ""})
    assert not _valid(schema, _out(grounds=[R102], status="supported") | {"explanation": ""})


def test_serialize_is_compact_like_constrained_generation():
    text = serialize(_example().target)
    assert ", " not in text.replace(json.dumps(_example().target.explanation)[1:-1], "")
    assert '"grounds":[' in text


def test_serialize_puts_prior_art_grounds_first():
    target = _example().target.model_copy(update={"evidence_status": "supported"})
    target.grounds = [Ground(**R112), Ground(**R103)]
    assert [g["statute"] for g in json.loads(serialize(target))["grounds"]] == ["103", "112"]


def test_parse_reports_json_and_schema_errors_without_repairing():
    assert parse('{"claim_mapping": [').error.startswith("json")
    two_refs = {"claim_mapping": [], "explanation": "x", "evidence_status": "supported",
                "grounds": [{"statute": "102", "references": ["1", "2"]}]}  # fmt: skip
    assert parse(json.dumps(two_refs)).error.startswith("schema")


def test_prompt_lists_limitations_and_cites_passages_by_id():
    example = _example()
    text = render_input(example.input)
    for i, limitation in enumerate(example.input.limitations, 1):
        assert f"{i}. {limitation}" in text
    for pid in example.input.passage_ids():
        assert f"[{pid}]" in text
    roles = [m["role"] for m in messages(example.input, shots=[(example.input, example.target)])]
    assert roles == ["system", "user", "assistant", "user"]
