# Project Specification

## Question

For legal reasoning over USPTO patent examination records, how does parameter-efficient fine-tuning (QLoRA SFT, then DPO) compare with automated prompt optimization (DSPy + GEPA)? And do the two complement each other?

## Task

The model plays the examiner. It sees one claim and the prior art the examiner had in front of them, and it decides what happens to that claim.

**Input**

- The claim text, plus the text of any claims it depends on
- The cited prior-art documents, as passages under a fixed context budget that is the same for every experiment (see [Evidence](#evidence))

The examiner's office action is **never** part of the input. It states the outcome ("Claims 1-5 are rejected under 35 U.S.C. 103"), so including it would leak the label.

**Output**

1. **Decision.** The grounds on which the claim is rejected, or none:
   - `102` (anticipated by a single reference)
   - `103` (obvious over one or more references)
   - `112` (indefinite or unsupported)
   - `101` (ineligible subject matter)

   A claim can have several grounds. An empty list means *no rejection*: the claim is allowable over the cited art.
2. **Claim mapping** (for 102/103). Each claim limitation, the reference and passages that disclose it, or a statement that no cited passage does.
3. **Explanation** of the decision in terms of that mapping.
4. **Evidence status.** One of:
   - `supported`: every limitation is mapped to cited text
   - `partially_supported`
   - `unsupported`
   - `not_applicable` (101/112 and no-rejection decisions)

   The model should abstain rather than assert a rejection the evidence doesn't support.

## Labels and negatives

Each example is one claim in an application's first non-final office action, where the pending claims are closest to the claims as filed.

- **Rejection grounds:** taken from the examiner's rejection statements in the office action text: the statute, plus the references the statement relies on. They're cross-checked against the Office Action Research Dataset (OARD), whose claim lists miss claims in about 18% of rejections ([feasibility.md](reports/feasibility.md)) and also fold in claims named in "as applied to claim 1 above". "Anticipated by or, in the alternative, obvious over X" gives both a 102 and a 103 ground.
- **No-rejection examples:** claims in the same office action that the examiner *allowed* or *objected to as allowable but dependent on a rejected claim*. The examiner judged these against the same cited art and found it didn't disclose or suggest the added limitations. They share the input distribution of the positives, so the decision can't be read off the context.
- **Reference targets:** the examiner's reasoning for that claim, with pin cites resolved to the cited passages. For 102/103 these are mapping and explanation targets. For 101/112, only the explanation.

Examiner statements are what the examiner asserted, not independently verified conclusions, and records keep that distinction. Generated text is never treated as ground truth.

**Claim text.** The claims come from the application's publication, or from the claims filed with it when it was never published. The text counts as verified when:

- the latest amendment before the action lists the claim as "(original)", or
- the examiner's restatement of the claim repeats at least half of its word 4-grams.

An amendment status such as "(currently amended)" or "(new)" marks the text as not what was examined. A loose restatement alone proves nothing, because examiners often argue by reference ("arguments analogous to claim 1 apply").

## Evidence

The prior art for a claim is every document cited in its office action: the documents OARD links to the action, plus those named in the rejections. Each document is split into its abstract and description paragraphs. The same documents go into every claim's input from that action, rejected or allowable.

Passages are chosen under a fixed budget of 2,500 tokens and 16 passages, with each passage capped at 300 tokens. With 4,096 tokens of context, this leaves room for the claim and its parents (549 tokens at the 95th percentile) and 1,000 tokens of instructions and output. There are two conditions:

- **`action_cited` (primary).** First come the passages the examiner pin-cited anywhere in the action, taking up to 70% of the budget. Then BM25 fills the rest, with the best two passages per limitation first, then the best for the claim as a whole. The cited pool is identical for every claim in the action, so it reveals what the examiner looked at, not what the examiner decided about this claim.
- **`retrieved`.** BM25 only. This is the realistic setting with no examiner involvement, kept as an ablation.

Ranking only ever uses the claim text, never the office action or the label. The examiner's own pin cites for a claim are used only to measure recall, and recall for both conditions is in [evidence_summary.json](reports/evidence_summary.json).

## Data sources

| Source | Use |
| --- | --- |
| USPTO Office Action Research Dataset (2017) | Rejection and allowance labels, cited documents |
| Open Data Portal file wrapper | Office action text (XML), claims as filed, amendment history |
| Open Data Portal full text (`APPXML-SPLT`, `PTGRXML-SPLT`) | Claims and cited prior-art text |

The cohort is non-final actions mailed 2008–2014. Paragraph-level evidence is available for US publications from 2005 onward. Older or granted patents, which are cited by column and line, and foreign references are evidence at the document level only. Details: [feasibility.md](reports/feasibility.md).

## Data contracts

The schemas live in `src/patent_reasoning/schema.py`, with JSON Schema exports in [docs/schemas/](schemas/). Each `Example` has three parts:

- `input`: what the model sees
- `target`: the reference output, with the same shape as the model's output
- `provenance`: source IDs, label sources and quality flags. Never shown to the model.

## Splits

| Partition | Target size |
| --- | --- |
| SFT train | ~20,000 claims |
| DPO preference pairs | ~3,000 |
| Validation | 1,000 |
| In-distribution test | 1,000 |
| OOD test (technology class / period) | 500–1,000 |

- Splits are disjoint by application family.
- Validation and test sets contain only fully grounded examples: claims as filed, and paragraph evidence for 102/103.
- The final test sets are never shown to any optimizer.

## Experimental matrix

| ID | Weights | Prompting |
| --- | --- | --- |
| A | Base, frozen | Zero-shot |
| B | Base, frozen | Handcrafted few-shot |
| C | Base, frozen | DSPy + GEPA |
| D | QLoRA SFT | Fixed |
| E | QLoRA SFT + DPO | Fixed |
| F | QLoRA SFT + DPO | DSPy + GEPA |

All variants use the same evidence retrieval and context budget, so differences can be attributed to the weights or the prompting.

## Metrics

- **Decision:** macro-F1 over {no rejection, 102, 103, 112, 101}, in-distribution and OOD. Also reported per class, including false-rejection and false-allowance rates.
- **Citations:** precision (cited passages exist and support the limitation) and unsupported-assertion rate.
- **Mapping:** limitation-level agreement with the examiner's pin cites, where available.
- **Cost:** inference latency and total experiment cost (GPU hours, optimizer calls, tokens).
- **Statistics:** bootstrap 95% confidence intervals and paired error comparisons.

## Model and training defaults

| Setting | Value |
| --- | --- |
| Base model | Qwen3-8B, thinking mode off for every variant |
| Quantization | 4-bit NF4 |
| LoRA rank / alpha | 16 / 32 |
| Learning rate | 2e-4 |
| Epochs | 2 |
| Max sequence length | 4,096 |

**Why Qwen3-8B.** All six variants share one base model, so what matters is that each gets a stable, fair run, not peak quality:

- QLoRA SFT, DPO and vLLM serving for GEPA all have well-tested support for it.
- Its size matches the compute plan.
- It's licensed under Apache 2.0.

Thinking mode stays off everywhere. Otherwise prompt-optimized variants could reason at length while fine-tuned ones don't, which would confound both accuracy and cost.

**Cross-model check.** If budget allows, the best fine-tuned variant and the best prompt-optimized variant are rerun on Gemma 4, to test whether the conclusion depends on the base model. The evidence budget is re-measured with each model's tokenizer; until then, tokens are estimated at 4 characters each.

## Reproducibility

- Dependencies are pinned in `uv.lock`.
- Each run is defined by a YAML file in `configs/`.
- Every run logs its config, seed and git commit to MLflow.
- Datasets ship with a manifest of source IDs and content hashes.

## Fallback scope

If grounded examples are too scarce, the project narrows to the decision task plus evidence verification. It prioritizes a smaller, trustworthy benchmark over a large, weakly labeled one.
