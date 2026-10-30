# Project Specification

## Question

For legal reasoning over USPTO patent examination records, how does parameter-efficient fine-tuning (QLoRA SFT, then DPO) compare with automated prompt optimization (DSPy + GEPA)? And do the two complement each other?

## Task

Given a patent claim, the office-action context and the cited prior-art passages, the model must:

1. **Classify the rejection basis**, e.g. 35 U.S.C. 101, 102, 103 or 112.
2. **Map claim limitations to evidence**, citing specific prior-art paragraphs.
3. **Explain the reasoning** in terms of that evidence.
4. **Report evidence status** (`supported`, `partially_supported`, `unsupported`). The model should abstain rather than assert a conclusion the evidence doesn't support.

## Data sources

| Source | Use |
| --- | --- |
| USPTO Office Action Research Dataset | Rejection labels, application IDs, prior-art citations |
| Office Action Text Retrieval API | Examiner office-action text |
| Patent Claims Research Dataset | Claim text and dependencies |
| Patent Examination Research Dataset (PatEx) | Application histories and metadata |
| USPTO Open Data Portal | Full text of cited prior-art publications |

The initial cohort is applications from 2008–2014, where the office-action and claims datasets overlap. Later filings are an optional temporal out-of-distribution (OOD) test.

## Record schema

```json
{
  "application_id": "...",
  "claim_number": 1,
  "claim_text": "...",
  "office_action_id": "...",
  "rejection_type": "103",
  "examiner_reasoning": "...",
  "prior_art": [{ "document_id": "...", "paragraph_id": "...", "text": "..." }],
  "target": {
    "legal_basis": "103",
    "claim_mapping": [],
    "explanation": "...",
    "evidence_status": "supported"
  }
}
```

Examiner-stated facts are kept separate from independently verified conclusions. Generated text is never treated as ground truth.

## Splits

| Partition | Target size |
| --- | --- |
| SFT train | ~20,000 |
| DPO preference pairs | ~3,000 |
| Validation | 1,000 |
| In-distribution test | 1,000 |
| OOD test (technology class / period) | 500–1,000 |

The splits are disjoint by application family. The final test sets are never shown to any optimizer.

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

- Rejection macro-F1, precision and recall, in-distribution and OOD
- Citation precision: whether cited paragraphs exist and support the claim
- Unsupported-assertion rate
- Inference latency and total experiment cost (GPU hours, optimizer calls, tokens)
- Bootstrap 95% confidence intervals and paired error comparisons

## Model and training defaults

| Setting | Value |
| --- | --- |
| Base model | 7B–8B instruct (Qwen or Llama class) |
| Quantization | 4-bit NF4 |
| LoRA rank / alpha | 16 / 32 |
| Learning rate | 2e-4 |
| Epochs | 2 |
| Max sequence length | 4,096 |

## Reproducibility

- Dependencies are pinned in `uv.lock`.
- Each run is defined by a YAML file in `configs/`.
- Every run logs its config, seed and git commit to MLflow.
- Datasets ship with a manifest of source IDs and content hashes.

## Fallback scope

If claim-to-evidence alignment turns out to be unreliable, the project narrows to rejection classification plus evidence verification. In that case it prioritizes a smaller, trustworthy benchmark over a large, weakly labeled one.
