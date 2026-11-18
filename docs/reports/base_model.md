# Base model and structured generation

**What this covers.** Loading the base model, and the generation path every variant shares: the prompt, constrained decoding, and parsing.

**Result.**

- **Model:** Qwen3-8B, thinking off, as the MLX 4-bit build (`mlx-community/Qwen3-8B-4bit`, revision `545dc42`, 4.6 GB). It runs locally on an M5 Pro with 24 GB of unified memory.
- **Deterministic:** greedy decoding gave byte-identical outputs on a second run for all 10 smoke-test examples.
- **Always well-formed:** constrained decoding gave 10 of 10 valid outputs, with no truncation.
- **Speed:** about 5 seconds per example on mains power, for zero-shot or few-shot. One pass over all three evaluation sets (4,418 examples) takes about 6 hours. On battery in Low Power Mode it is several times slower.

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
- each ground has the right number of references for its statute: one for 102, at least one for 103, none for 101 and 112
- lists and strings are capped, so a run can't loop until the token limit
- the JSON is compact, the same as `serialize`

Document and passage IDs aren't constrained. Citing something outside the input stays possible, so the faithfulness evaluator can measure it.

**Parsing.** The evidence status must match the decision (`not_applicable` exactly when there is no 102/103 ground). The MLX runs below checked this after generation and recorded breaches as invalid, without repair. The schema now enforces it (see the update at the end), and `parse` still checks every rule.

## What the smoke test showed

Three problems came up in the first runs, and the design above is the result:

- **Greedy decoding loops on free-form mappings.** With the mapping as a free list, 4 of 10 outputs repeated the same limitations until they hit the 1,024-token limit. A repetition penalty would also penalize the verbatim quoting a mapping needs. Listing the claim's limitations in the input, and making the schema one entry per limitation, removed the loops. The limitations are the ones the pipeline already splits and the targets already quote, so the input carries no new information, and mappings can now be scored limitation by limitation. This added `input.limitations` (schema 1.3).
- **Without a format example, the model gave up on the mapping.** Unconstrained, it wrote the mapping as an object keyed by limitation. Under the constraint, greedy decoding then chose the empty list. The one-line shape example in the system message fixed this.
- **Grammar compile time.** The default outlines backend took 2 minutes to compile the schema. llguidance takes under a second and gave identical outputs.

The 10 examples aren't a measurement of accuracy. For what it's worth, the zero-shot model got 4 of 10 decisions right and leaned toward 103.

## Baselines

The two frozen-model variants share everything except the demonstrations:

| | Zero-shot (A) | Few-shot (B) |
| --- | --- | --- |
| Config | [base_zero_shot.yaml](../../configs/generation/base_zero_shot.yaml) | [base_few_shot.yaml](../../configs/generation/base_few_shot.yaml) |
| Before each example | the system message | the system message and 5 demonstrations |
| Shared prefix (cached) | 549 tokens | 6,414 tokens |
| Seconds per example, median of 20 | 5.6 | 4.3 |
| Valid outputs, of 20 | 19 | 19 |

**Demonstrations.** One per decision class, all from train, so no evaluation example can be one:
- 102: GU8VZQF9:10, where one limitation is disclosed by Mitchell [0042]
- 103: HV6MQ40W:6, where the primary reference lacks a feature and Shin [0056] supplies it
- 112: GLP467AE:15, where two terms lack antecedent basis
- 101: G291HJC2:7, a process not tied to a machine or transformation
- no rejection: I4A5Z69Q:2, where the reasons for allowance name the limitation

Each was read in full. Each was chosen because its target is fully grounded, it is short, and its explanation is the examiner's reasoning about that claim itself. No train example with no rejection has both an explanation and a full mapping; the one chosen maps its single limitation as not disclosed, which the examiner's reasons state.

**Prefix cache.** The system message and demonstrations are the same for every example, so their KV cache is computed once (`MLXGenerator.set_prefix`). Each generation processes only the example's own message, then the cache is trimmed back. On zero-shot, cached outputs are byte-identical to uncached ones. With the cache, few-shot costs no more than zero-shot, despite prompts of about 9,000 tokens.

The two invalid outputs both have an evidence status that contradicts their grounds. Scoring reads the decision from the grounds and reports validity separately.

## Tokens

Measured with the Qwen3 tokenizer ([prompt_tokens.json](../results/prompt_tokens.json)):

| | Train | Validation | Test | OOD |
| --- | --- | --- | --- | --- |
| Prompt tokens, median / p95 | 3,235 / 3,703 | 3,243 / 3,755 | 3,250 / 3,705 | 3,240 / 4,241 |
| Target tokens, median / p95 | 238 / 614 | 227 / 580 | 214 / 581 | 188 / 695 |
| Prompt + target over 4,096 | 933 (4.5%) | 65 (4.1%) | 68 (4.2%) | 240 (20%) |

- **The estimate held up.** Prompts average 4.3 characters per token, against the estimate of 4 used to set the evidence budget.
- **Inference has room.** Qwen3's context is 32k tokens, and the longest prompt is 18,129 tokens (an OOD claim). Generation doesn't truncate anything.
- **Training is the constraint.** At a 4,096-token sequence length, 4.5% of train examples don't fit. OOD is worse because of long chemistry and biotech claims. How to handle these (truncate evidence, drop, or raise the length) is decided with the SFT config.

## Open points

- **Backend and quantization for the comparison.** Resolved: every variant is evaluated on a CUDA machine with vLLM and bf16 weights. See the update below.
- **Train targets with partial mappings.** Resolved: art rejections without a full mapping are excluded from SFT (`patent_reasoning.train.sft_records`).

## Update: the CUDA backend

The comparison runs on a rented A100 80 GB with vLLM (`VLLMGenerator`), Qwen3-8B in bf16 at revision `b968826`. vLLM 0.34 no longer loads bitsandbytes NF4, so adapters trained against the NF4 base are applied to the bf16 weights for evaluation, for every variant alike.

Three changes came with the move:

- **The evidence-status rule is in the schema.** On the first GPU check, 42% of outputs were invalid: in bf16 the model often marks an allowance "unsupported", where the 4-bit model wrote "not_applicable". The schema is now one of three shapes that differ after the explanation: no grounds and "not_applicable"; only 112/101 grounds and "not_applicable"; or a 102/103 ground first and a support level. The cost is one convention, that a prior-art ground is listed first, and `serialize` writes targets and demonstrations that way. After the change, 200 of 200 outputs were valid.
- **An allowance may have an empty explanation.** Examiners rarely give reasons for an allowable claim, so most such SFT targets have none. They keep an empty explanation rather than an invented one, and only the no-rejection shape allows it.
- **`serialize` is compact.** It had written JSON with spaces after separators, while constrained generation writes none, so few-shot demonstrations didn't match the model's own output format. Targets and demonstrations are now compact too.

Compact output is set both in the engine (`disable_any_whitespace`) and in the schema (`x-guidance`); the per-request setting alone was ignored and produced pretty-printed JSON. Throughput is about 0.35 seconds per example in batches of 256, against 5–10 seconds on the laptop.
