"""The prompt every variant starts from: a fixed system message and the example input
rendered as text.

Passages are listed under their ``passage_id``, which is what the model cites, so a citation
can be checked against the input exactly. Prompt-optimized variants may rewrite the system
message, but the rendering of the input stays fixed.
"""

from __future__ import annotations

from patent_reasoning.model.output import serialize
from patent_reasoning.schema import ExampleInput, ModelOutput

SYSTEM = """\
You are a USPTO patent examiner writing the first office action for one claim. You are given \
the claim, the claims it depends on, and passages from the prior art cited in the \
application. Decide whether the claim is rejected, and on what grounds.

Grounds:
- "102": every limitation of the claim, read with its parent claims, is disclosed by a single \
reference. Exactly one reference.
- "103": the claim would have been obvious over one or more references, alone or combined. At \
least one reference.
- "112": the claim is indefinite (for example, a term without antecedent basis or a scope that \
can't be determined) or lacks written description or enablement. No references.
- "101": the claim is directed to ineligible subject matter (an abstract idea, law of nature \
or natural phenomenon without significantly more, or software or a signal per se). No \
references.
A claim can have several grounds. An empty list of grounds means no rejection: the claim is \
allowable over the cited art. Reject only when the passages support it; if they don't show a \
limitation, say so rather than assume it.

Answer with one JSON object, fields in this order:
- "claim_mapping": for a 102 or 103 rejection, or for a claim you find allowable over the art, \
one entry per listed limitation of this claim, in order, quoting it exactly. For each, \
"disclosed" is whether the cited art discloses or suggests it; "document_id" is the reference \
that does (null if none); "passage_ids" are the passages that show it. Use [] for a 101 or 112 \
rejection only.
- "explanation": your reasoning for the decision, in terms of the mapping and the passages.
- "grounds": the rejection grounds, each with "statute" and "references" (document IDs).
- "evidence_status": for a 102 or 103 rejection, "supported" if every limitation is mapped to \
a passage, "partially_supported" if some are, "unsupported" if none are; otherwise \
"not_applicable".

Cite only document IDs and passage IDs that appear in the input. The shape, with placeholder \
values:
{"claim_mapping":[{"limitation":"<text of the claim>","disclosed":true,"document_id":"<document \
ID>","passage_ids":["<passage ID>"]}],"explanation":"<reasoning>","grounds":[{"statute":"103",\
"references":["<document ID>"]}],"evidence_status":"supported"}"""


def render_input(inp: ExampleInput) -> str:
    lines = [f"Claim {inp.claim.number}:", inp.claim.text.strip(), "", "Limitations of this claim:"]
    lines += [f"{i}. {lim}" for i, lim in enumerate(inp.limitations, 1)] or ["(none of its own)"]
    if inp.parent_claims:
        lines += ["", "Parent claims (nearest first):"]
        lines += [f"Claim {c.number}: {c.text.strip()}" for c in inp.parent_claims]
    lines += ["", "Cited prior art:"]
    for doc in inp.prior_art:
        title = f": {doc.title}" if doc.title else ""
        lines.append(f"\nDocument {doc.document_id} ({doc.kind.replace('_', ' ')}){title}")
        if not doc.passages:
            lines.append("(no text available)")
        for p in doc.passages:
            lines.append(f"[{p.passage_id}] {p.text.strip()}")
    return "\n".join(lines)


def prefix(
    system: str = SYSTEM, shots: list[tuple[ExampleInput, ModelOutput]] = ()
) -> list[dict[str, str]]:
    """The messages every example shares: the system message, then any demonstrations."""
    out = [{"role": "system", "content": system}]
    for shot_input, shot_output in shots:
        out.append({"role": "user", "content": render_input(shot_input)})
        out.append({"role": "assistant", "content": serialize(shot_output)})
    return out


def messages(
    inp: ExampleInput,
    system: str = SYSTEM,
    shots: list[tuple[ExampleInput, ModelOutput]] = (),
) -> list[dict[str, str]]:
    """Chat messages for one example, with optional demonstrations before it."""
    return [*prefix(system, shots), {"role": "user", "content": render_input(inp)}]
