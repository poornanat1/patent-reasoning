# Patent Legal Reasoning

Can an open 8B model learn to reason like a patent examiner? And is it better to teach it by fine-tuning or by optimizing its prompt?

This project builds a benchmark from real USPTO examination records and uses it to compare two ways of improving Qwen3-8B: fine-tuning (QLoRA SFT, then DPO) and prompt optimization (DSPy + GEPA).

## The task

The model is given one patent claim and the prior art the examiner cited. It decides whether the claim is rejected, and on which ground: 102 (anticipated), 103 (obvious), 112 (indefinite) or 101 (ineligible). It then maps each limitation of the claim to the passage of prior art that discloses it, and explains its decision. The examiner's office action is never part of the input.

## Dataset

The dataset has about 25,000 claims from real office actions, split by patent family into train, validation, test and an out-of-distribution test set (biotech and chemistry). Two manual audits checked the labels.

## Results so far

| Frozen Qwen3-8B | Macro-F1 (test) | Macro-F1 (OOD) |
| --- | --- | --- |
| Zero-shot | 0.276 | 0.162 |
| Few-shot | 0.258 | 0.147 |

The base model usually finds the passage the examiner relied on, but it often confuses 102 with 103, and it almost never predicts 112 or 101. Fine-tuning is next.

## More

- [Design](docs/SPEC.md)
- [Reports](docs/README.md)
