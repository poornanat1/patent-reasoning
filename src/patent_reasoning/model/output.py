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

Rules the schema can't express (a 102 ground has exactly one reference, the evidence status
matches the decision) are checked by ``parse``: an output that breaks them is invalid, not
repaired.
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
    defs["Ground"]["properties"]["references"]["maxItems"] = MAX_REFERENCES
    defs["Ground"]["required"] = ["statute", "references"]
    for p in [*defs["Ground"]["properties"].values(), *schema["properties"].values()]:
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
