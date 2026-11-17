"""SFT records: the prompt every variant sees, and the target the model should write.

The prompt is the zero-shot prompt (system message and the rendered input) under the chat
template with thinking off, exactly as generation builds it. The completion is the target
serialized as the model writes it (``serialize``), so training and generation see one
format, and it satisfies the example's generation schema.

A training example becomes a record unless:

- **an art rejection's mapping is empty or partial**: a 102/103 target must tie every
  limitation to a passage. Filling the gaps with "disclosed, no passage" would teach the
  uncited assertions the citation metrics penalise.
- **a rejection has no explanation**: there is nothing to learn the reasoning from.
- **it doesn't fit**: prompt and completion together exceed ``max_tokens``. Excluded rather
  than truncated, so no target is cut off.

An allowable claim with no examiner explanation (most of them: examiners rarely give
reasons) keeps an empty explanation rather than an invented one. Dropping those would leave
"no rejection" at 5% of training against 18% of train and 23-32% of the evaluation sets.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable

from patent_reasoning.model.output import serialize
from patent_reasoning.model.prompt import messages
from patent_reasoning.schema import NO_REJECTION, PRIOR_ART_STATUTES, Example, ModelOutput


def target(example: Example) -> tuple[ModelOutput | None, str | None]:
    """The SFT target for ``example``, or None and the reason it is excluded."""
    t = example.target
    if any(g.statute in PRIOR_ART_STATUTES for g in t.grounds):
        mapped = [m.limitation for m in t.claim_mapping]
        if not mapped:
            return None, "art_mapping_empty"
        if mapped != example.input.limitations:
            return None, "art_mapping_partial"
    if t.explanation is None:
        if t.decision != {NO_REJECTION}:
            return None, "rejection_without_explanation"
        t = t.model_copy(update={"explanation": ""})
    if t.claim_mapping and [m.limitation for m in t.claim_mapping] != example.input.limitations:
        return None, "mapping_not_input_limitations"
    return t, None


def records(
    examples: Iterable[Example],
    render: Callable[[list[dict[str, str]]], str],
    count_tokens: Callable[[str], int],
    max_tokens: int,
    eos: str,
) -> tuple[list[dict], Counter]:
    """Prompt/completion records and a tally of exclusions. ``render`` applies the chat
    template with a generation prompt; ``eos`` ends each completion."""
    out, excluded = [], Counter()
    for e in examples:
        t, reason = target(e)
        if t is None:
            excluded[reason] += 1
            continue
        prompt = render(messages(e.input))
        completion = serialize(t) + eos
        if count_tokens(prompt) + count_tokens(completion) > max_tokens:
            excluded["over_max_tokens"] += 1
            continue
        out.append({"example_id": e.example_id, "prompt": prompt, "completion": completion,
                    "decision": "+".join(sorted(t.decision))})  # fmt: skip
    return out, excluded
