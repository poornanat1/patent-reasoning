# Patent Legal Reasoning

Can an open 8B model learn to reason like a patent examiner? And is it better to teach it by fine-tuning or by optimizing its prompt?

This project builds a benchmark from real USPTO examination records and uses it to compare two ways of improving Qwen3-8B: fine-tuning (QLoRA SFT, then DPO) and prompt optimization (DSPy + GEPA).

## The task

The model is given one patent claim and the prior art the examiner cited. It decides whether the claim is rejected, and on which ground: 102 (anticipated), 103 (obvious), 112 (indefinite) or 101 (ineligible). It then maps each limitation of the claim to the passage of prior art that discloses it, and explains its decision. The examiner's office action is never part of the input.

## Dataset

The dataset has about 25,000 claims from real office actions, split by patent family into train, validation, test and an out-of-distribution test set (biotech and chemistry). Two manual audits checked the labels.

## Results so far

Validation macro-F1 over the five decision classes:

| Qwen3-8B | Macro-F1 (val) |
| --- | --- |
| Zero-shot | 0.241 |
| Few-shot | 0.248 |
| QLoRA SFT | 0.455 |

Fine-tuning nearly doubles the score of the frozen model. It cuts false allowances from 27% to 10%, and it learns to predict 112. Its main remaining errors are calling a 102 a 103 and missing 101. DPO is next.

## More

- [Design](docs/SPEC.md)
- [Reports](docs/README.md)
