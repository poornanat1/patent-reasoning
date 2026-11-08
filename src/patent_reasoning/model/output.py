"""The JSON the model generates, and how it is read back into a ``ModelOutput``.

Generation follows ``FIELD_ORDER``: the claim mapping first, then the explanation, then the
grounds. Thinking mode is off, so the JSON is the only place the model can reason before it
commits to a decision. Few-shot demonstrations and SFT targets are serialized in the same
order (``serialize``), so every variant sees one format.

The generation schema is ``ModelOutput``'s, with every field required, the explanation a
string, and caps on list and string lengths so a run can't loop until ``max_new_tokens``.
It is built per example: the claim mapping is either empty or one entry per limitation of
the input, in order, each quoting its limitation exactly. Without that, greedy decoding
repeats limitations until it runs out of tokens. Document and passage IDs are not
constrained, so citing something outside the input stays possible and measurable.

Each statute's reference count is in the schema too: one for 102, at least one for 103,
none for 101 and 112. The rule the schema can't express, that the evidence status matches
the decision, is checked by ``parse``: an output that breaks it is invalid, not repaired.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

from pydantic import ValidationError

from patent_reasoning.schema import ModelOutput

FIELD_ORDER = ("claim_mapping", "explanation", "grounds", "evidence_status")
MAX_PASSAGES_PER_LIMITATION = 16  # the whole evidence budget
MAX_REFERENCES = 6
MAX_GROUNDS = 4
MAX_EXPLANATION_CHARS = 2400


def generation_schema(limitations: list[str]) -> dict:
    """JSON Schema for constrained decoding of one example (fields in ``FIELD_ORDER``, all
    required; the mapping empty or one entry per limitation, in order)."""
    schema = copy.deepcopy(ModelOutput.model_json_schema())
    defs = schema["$defs"]
    props = schema["properties"]
    props["explanation"] = {"type": "string", "minLength": 1, "maxLength": MAX_EXPLANATION_CHARS}
    props["grounds"]["maxItems"] = MAX_GROUNDS

    mapping = defs.pop("LimitationMapping")
    mapping["properties"]["passage_ids"]["maxItems"] = MAX_PASSAGES_PER_LIMITATION
    mapping["required"] = ["limitation", "disclosed", "document_id", "passage_ids"]
    entries = [copy.deepcopy(mapping) for _ in limitations]
    for entry, limitation in zip(entries, limitations, strict=True):
        entry["properties"]["limitation"] = {"const": limitation}
    empty = {"type": "array", "maxItems": 0}
    full = {"type": "array", "prefixItems": entries, "items": False,
            "minItems": len(entries), "maxItems": len(entries)}  # fmt: skip
    props["claim_mapping"] = {"anyOf": [empty, full]} if limitations else empty

    schema["properties"] = {k: props[k] for k in FIELD_ORDER}
    schema["required"] = list(FIELD_ORDER)
    del schema["$defs"]  # Ground and LimitationMapping are inlined above and below
    refs = {"type": "array", "items": {"type": "string", "minLength": 1}}
    counts = {"102": (1, 1), "103": (1, MAX_REFERENCES), "101": (0, 0), "112": (0, 0)}
    props["grounds"]["items"] = {"anyOf": [
        {"type": "object", "additionalProperties": False, "required": ["statute", "references"],
         "properties": {"statute": {"const": statute},
                        "references": refs | {"minItems": lo, "maxItems": hi}}}
        for statute, (lo, hi) in counts.items()
    ]}  # fmt: skip
    for p in schema["properties"].values():
        p.pop("default", None)
    for entry in entries:
        for p in entry["properties"].values():
            p.pop("default", None)
    return schema


def serialize(output: ModelOutput) -> str:
    """``output`` as the model should write it: compact, fields in ``FIELD_ORDER``."""
    data = output.model_dump(mode="json")
    data = {k: data[k] for k in FIELD_ORDER}
    return json.dumps(data, ensure_ascii=False)


@dataclass
class Parsed:
    output: ModelOutput | None
    error: str | None  # "json": not JSON; "schema": breaks a ModelOutput rule


def parse(text: str) -> Parsed:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return Parsed(None, f"json: {e}")
    try:
        return Parsed(ModelOutput.model_validate(data), None)
    except ValidationError as e:
        first = e.errors()[0]
        return Parsed(None, f"schema: {first['msg']} at {'.'.join(map(str, first['loc']))}")
