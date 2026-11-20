# Docs

- [SPEC.md](SPEC.md): the full design. It covers the task, labels, evidence, splits, the experiment matrix, metrics and training defaults.
- [reproducing.md](reproducing.md): every command, from downloading the data to training.
- [schemas/](schemas/): the data contracts (`Example`, `ModelOutput`), as JSON Schema with two worked examples.

## Reports

Each report opens with what it covers and its result. They are listed in the order the work was done.

**Data**

1. [oard_audit_findings.md](reports/oard_audit_findings.md): what the USPTO Office Action Research Dataset contains, and how far its labels can be trusted
2. [feasibility.md](reports/feasibility.md): whether labels, office action text, claims and prior art can be joined into traceable examples
3. [claim_level_dataset.md](reports/claim_level_dataset.md): one record per claim, and the evidence passages each input shows
4. [sft_corpus.md](reports/sft_corpus.md): which records become training examples, and why the rest are excluded
5. [splits_and_preferences.md](reports/splits_and_preferences.md): family-disjoint splits and DPO preference pairs
6. [dataset_v1.md](reports/dataset_v1.md): two manual audits, the fixes they led to, and the frozen dataset v1

**Models**

7. [base_model.md](reports/base_model.md): the base model, the prompt, and constrained generation
8. [evaluation.md](reports/evaluation.md): the scoring rubric, fixed before any training
9. [baselines.md](reports/baselines.md): zero-shot and few-shot Qwen3-8B, with a review of 50 failures
10. [sft.md](reports/sft.md): the QLoRA SFT adapter on validation, decision thresholds, and a review of 50 failures

The first three reports were written on early samples. The banners at the top of each one point to the current numbers.

## Raw outputs

- [results/](results/): machine-written summaries the pipeline regenerates, including the dataset v1 manifest
- [audits/](audits/): the hand-labeled audit samples and verdicts, and the failure reviews of each variant. The split builder reads the verdicts to drop audited errors from evaluation, so treat them as inputs, not just records.
