# Base model and structured generation

**What this covers.** Loading the base model, and the generation path every variant shares: the prompt, constrained decoding, and parsing.

**Result.**

- **Model:** Qwen3-8B, thinking off, as the MLX 4-bit build (`mlx-community/Qwen3-8B-4bit`, revision `545dc42`, 4.6 GB). It runs locally on an M5 Pro with 24 GB of unified memory.
- **Deterministic:** greedy decoding gave byte-identical outputs on a second run for all 10 smoke-test examples.
- **Always well-formed:** constrained decoding gave 10 of 10 valid outputs, with no truncation.
- **Speed:** about 12 seconds per example for a 3,200-token prompt. One pass over validation (1,597 examples) takes about 5 hours; over all three evaluation sets (4,418), about 15.

Code: `src/patent_reasoning/model/` (prompt, output schema, generator), `scripts/generate.py`, `scripts/measure_prompt_tokens.py`. Config: [configs/generation/base_zero_shot.yaml](../../configs/generation/base_zero_shot.yaml).

## Generation

**Prompt.** A fixed system message describes the task, the four grounds, the output fields and the rules for references, with a one-line example of the JSON shape. The user message renders the input:
- the claim
- its numbered limitations
- its parent claims
- each cited document, with its passages listed under their passage IDs

The model cites passages by those IDs, so every citation can be checked against the input exactly.

**Field order.** The model writes the claim mapping first, then the explanation, then the grounds and the evidence status. With thinking off, the JSON is the only place it can reason before committing to a decision. Few-shot demonstrations and SFT targets use the same order (`serialize`).

**Constrained decoding.** Each example gets its own JSON schema, compiled by llguidance through outlines in under a second:
- every field is required, and the explanation is a string
- the mapping is either empty, or exactly one entry per listed limitation, in order, each quoting its limitation
- lists and strings are capped, so a run can't loop until the token limit
- the JSON is compact, the same as `serialize`

Document and passage IDs aren't constrained. Citing something outside the input stays possible, so the faithfulness evaluator can measure it.

**Parsing.** Some rules can't be expressed in the schema: a 102 ground needs exactly one reference, and the evidence status must match the decision. An output that breaks one of them is recorded as invalid with the reason. It isn't repaired or retried.

## What the smoke test showed

Three problems came up in the first runs, and the design above is the result:

- **Greedy decoding loops on free-form mappings.** With the mapping as a free list, 4 of 10 outputs repeated the same limitations until they hit the 1,024-token limit. A repetition penalty would also penalize the verbatim quoting a mapping needs. Listing the claim's limitations in the input, and making the schema one entry per limitation, removed the loops. The limitations are the ones the pipeline already splits and the targets already quote, so the input carries no new information, and mappings can now be scored limitation by limitation. This added `input.limitations` (schema 1.3).
- **Without a format example, the model gave up on the mapping.** Unconstrained, it wrote the mapping as an object keyed by limitation. Under the constraint, greedy decoding then chose the empty list. The one-line shape example in the system message fixed this.
- **Grammar compile time.** The default outlines backend took 2 minutes to compile the schema. llguidance takes under a second and gave identical outputs.

The 10 examples aren't a measurement of accuracy. For what it's worth, the zero-shot model got 4 of 10 decisions right and leaned toward 103.

## Tokens

Measured with the Qwen3 tokenizer ([prompt_tokens.json](prompt_tokens.json)):

| | Train | Validation | Test | OOD |
| --- | --- | --- | --- | --- |
| Prompt tokens, median / p95 | 3,235 / 3,703 | 3,243 / 3,755 | 3,250 / 3,705 | 3,240 / 4,241 |
| Target tokens, median / p95 | 238 / 614 | 227 / 580 | 214 / 581 | 188 / 695 |
| Prompt + target over 4,096 | 933 (4.5%) | 65 (4.1%) | 68 (4.2%) | 240 (20%) |

- **The estimate held up.** Prompts average 4.3 characters per token, against the estimate of 4 used to set the evidence budget.
- **Inference has room.** Qwen3's context is 32k tokens, and the longest prompt is 18,129 tokens (an OOD claim). Generation doesn't truncate anything.
- **Training is the constraint.** At a 4,096-token sequence length, 4.5% of train examples don't fit. OOD is worse because of long chemistry and biotech claims. How to handle these (truncate evidence, drop, or raise the length) is decided with the SFT config.

## Open points

- **Backend and quantization for the comparison.** MLX 4-bit (affine, group size 64) isn't the NF4 that QLoRA trains on. Every variant must be evaluated on the same backend and quantization, and that's decided before the SFT phase. Either train LoRA adapters with MLX on this 4-bit model, or evaluate everything on a CUDA machine with NF4. The generator interface is backend-neutral, and outlines supports vLLM and transformers, so a CUDA backend is a small addition.
- **Train targets with partial mappings.** Generation allows only an empty or full mapping, but 31% of train's 102/103 targets have no mapping and others are partial. They'll be handled with the SFT config.
