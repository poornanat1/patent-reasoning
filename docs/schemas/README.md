# Data contracts

One `Example` per claim per office action. The models are defined in [`src/patent_reasoning/schema.py`](../../src/patent_reasoning/schema.py), and the JSON Schemas here are generated from them:

```bash
uv run python scripts/export_schemas.py          # regenerate
uv run python scripts/export_schemas.py --check  # fail if stale (also covered by tests)
```

| File | What it describes |
| --- | --- |
| [example.schema.json](example.schema.json) | A full example: `input`, `target`, `provenance` |
| [model_output.schema.json](model_output.schema.json) | What the model must produce; also the shape of `target` |
| [examples/rejection_102.json](examples/rejection_102.json) | Claim 2 of application 13/624,988, rejected under 102 over Hallak (US 2013/0173955), with the examiner's pin cite resolved to paragraph [0131] |
| [examples/allowable_dependent.json](examples/allowable_dependent.json) | Claim 4 of the same office action, which the examiner found allowable over the same art |

## The three parts

- **`input`** is everything the model sees: the claim, its own limitations as split by the pipeline, the claims it depends on (nearest first), and the cited prior art as passages. It never contains the office action, because the action states the outcome. Unknown fields are rejected, so nothing extra can slip in.
- **`target`** is a `ModelOutput`:
  - `grounds`: the rejection grounds, empty for no rejection
  - `claim_mapping`: this claim's limitations mapped to passages
  - `explanation`
  - `evidence_status`
- **`provenance`** records where each part came from and how far to trust it. It is used for filtering, splits and audits, never as model input.

## Conventions

- **Decision labels.** `ModelOutput.decision` is the set of statutes, or `{"no_rejection"}`. Scoring uses these five classes: no rejection, 102, 103, 112, 101. A claim can have several grounds, including two 103 grounds over different combinations.
- **Ground references.** A 102 ground relies on exactly one reference, a 103 ground on one or more, and 101/112 on none.
- **Evidence status** applies only when there is a 102/103 ground. Otherwise it is `not_applicable`.
- **Passage IDs** are `<document_id>:<locator key>`, for example `20130173955:0131`. Every passage or document that a target cites must appear in the input.
- **Claim mapping** covers only the claim's own limitations. A dependent claim's parent limitations are mapped in the parent's example. Limitations are quoted verbatim from the claim, as listed in `input.limitations`. A model's mapping is either empty or one entry per listed limitation, in order; a training target may list only the limitations the examiner's cites reach (`mapping_partial`).
- **No-rejection examples** come from the office action text. Claims are either "allowed" or "objected to as being dependent upon a rejected base claim, but would be allowable if rewritten…". They don't come from OARD's `objected` rows, which also include objections for informalities. The examiner judged these claims against the same cited art.
- **Explanations for allowable claims.** When the action has a statement of reasons for allowable subject matter that names the claim, it is the explanation (`explanation_source: "reasons_for_allowance"`).
- **Missing explanations.** When the examiner gives no reasons (common for allowable claims), `target.explanation` is `null`, `provenance.explanation_source` is `"none"`, and the example is flagged `no_reference_explanation`.
- **Evidence condition.** `provenance.evidence_condition` records how the input passages were chosen: `action_cited` (primary) or `retrieved`. See the [spec](../SPEC.md#evidence).
- **Examiner text** is kept verbatim in `provenance.examiner_text`. It records what the examiner asserted, not a verified conclusion.
