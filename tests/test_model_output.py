import json
from pathlib import Path

from patent_reasoning.model.output import FIELD_ORDER, generation_schema, parse, serialize
from patent_reasoning.model.prompt import messages, render_input
from patent_reasoning.schema import Example


def _example(name: str = "rejection_102.json") -> Example:
    return Example.model_validate_json(Path(f"docs/schemas/examples/{name}").read_text())


def test_serialize_follows_field_order_and_round_trips():
    target = _example().target
    text = serialize(target)
    assert list(json.loads(text)) == list(FIELD_ORDER)
    assert parse(text).output == target


def test_schema_mapping_is_empty_or_one_entry_per_limitation_in_order():
    limitations = ["a sensor", "a valve"]
    mapping = generation_schema(limitations)["properties"]["claim_mapping"]
    empty, full = mapping["anyOf"]
    assert empty["maxItems"] == 0
    assert [e["properties"]["limitation"]["const"] for e in full["prefixItems"]] == limitations
    assert full["minItems"] == full["maxItems"] == 2 and full["items"] is False
    assert generation_schema([])["properties"]["claim_mapping"] == {"type": "array", "maxItems": 0}


def test_schema_requires_every_field_in_order():
    schema = generation_schema(["a sensor"])
    assert list(schema["properties"]) == schema["required"] == list(FIELD_ORDER)
    assert schema["properties"]["explanation"]["type"] == "string"


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
