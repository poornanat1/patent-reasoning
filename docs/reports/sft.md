# SFT: validation scores and errors

**What this covers.** This report evaluates experiment D, the QLoRA SFT adapter, on validation. It compares the adapter with the two GPU baselines ([baselines.md](baselines.md)), tests per-class decision thresholds over sampled decisions, and reviews 50 failures. It covers validation only. The held-out test and OOD evaluation of the frozen adapter comes later, so no choice made here has looked at those scores.

**Result.**

- **SFT is far ahead of both baselines.** Validation macro-F1 is 0.455 (95% interval 0.406–0.496). That is +0.214 over zero-shot (0.165 to 0.256) and +0.207 over few-shot (0.161 to 0.250). In every bootstrap resample, SFT is ahead.
- **Every class improves, but 101 barely.** 112 goes from F1 0 to 0.44 and 101 from 0 to 0.05. False rejections drop from 50% to 34%, and false allowances from 27% to 10%.
- **Greedy decoding is the bottleneck for rare and combined grounds.** Per-class thresholds over 16 sampled decisions raise macro-F1 to 0.546 in a cross-validated estimate (+0.091, 0.056 to 0.125), and 101 F1 to 0.36.
- **The main remaining error is calling a 102 a 103.** All 153 of these errors map only one reference, so the second 103 reference is named but never cited.
- **The model never abstains.** It marks every one of its 1,114 art rejections "supported", because every art-rejection target it was trained on says "supported".

Pipelines:

- `scripts/score.py configs/generation/gpu_sft.yaml --split val --compare <baseline>`
- `scripts/sample_decisions.py` and `scripts/tune_thresholds.py fit` (thresholds in [sft_thresholds.json](../results/sft_thresholds.json))
- `scripts/sample_failures.py` (review in [gpu_sft_val_review.csv](../audits/failures/gpu_sft_val_review.csv))

## The adapter

Trained with [configs/training/sft.yaml](../../configs/training/sft.yaml): QLoRA on Qwen3-8B in 4-bit NF4, rank 16 and alpha 32 on every linear projection, learning rate 2e-4, 2 epochs at an effective batch of 16 (1,760 steps, 14.4 hours on one A100).

The training records are the 14,070 train examples whose art rejections are fully mapped and which fit in 4,096 tokens. 6,395 were dropped for empty or partial art mappings, and 444 for length ([sft_records.py](../../src/patent_reasoning/train/sft_records.py)). Allowable claims without examiner reasons are kept, with an empty explanation.

For evaluation the adapter runs on the bf16 weights in vLLM, with the zero-shot prompt and the same grammar as the baselines ([configs/generation/gpu_sft.yaml](../../configs/generation/gpu_sft.yaml)).

**Checkpoint choice.** Validation loss was lowest at step 400 (0.507) and rose to 0.55 during epoch 2. Even so, the final adapter beats checkpoint 900 on validation macro-F1 (0.455 against 0.423). Loss tracks the long explanation text more than the decision, so checkpoints are selected on macro-F1, not loss.

## Against the baselines

| Validation | Zero-shot | Few-shot | SFT |
| --- | --- | --- | --- |
| Macro-F1 | 0.241 | 0.248 | **0.455** |
| 95% interval | 0.216–0.263 | 0.222–0.271 | 0.406–0.496 |
| Exact match | 0.320 | 0.354 | **0.504** |
| False-rejection rate | 0.503 | 0.373 | **0.337** |
| False-allowance rate | 0.265 | 0.317 | **0.098** |
| Valid outputs | **100.0%** | 99.8% | 98.1% |
| Citation precision | 0.910 | **0.914** | 0.888 |
| Art rejections citing nothing | 0.008 | 0.007 | **0.005** |
| Examiner pin recall | 0.373 | 0.321 | **0.448** |

**Per-class F1** (predictions in brackets; support: no rejection 362, 102 564, 103 476, 112 309, 101 118):

| Class | Zero-shot | Few-shot | SFT |
| --- | --- | --- | --- |
| No rejection | 0.414 (507) | 0.458 (616) | **0.655** (356) |
| 102 | 0.384 (384) | 0.520 (721) | **0.573** (574) |
| 103 | 0.406 (706) | 0.244 (253) | **0.564** (542) |
| 112 | 0.000 (1) | 0.000 (0) | **0.436** (99) |
| 101 | 0.000 (1) | 0.017 (3) | **0.048** (8) |

SFT gets 442 validation examples right that zero-shot gets wrong, and 148 the other way round. Citations hold up: precision falls slightly (0.91 to 0.89), while recall of the examiner's own pin cites rises from 0.37 to 0.45.

**Not yet good enough on 112 and 101.** 112 predictions are precise (0.90) but recall only 0.29. 101 is predicted 8 times against 118 gold claims, even though 1,071 training records carry a 101 ground. Training shifted the decision prior, but greedy decoding still takes the single most likely ground, and a second ground is almost never it: only 11 of 1,597 outputs have two or more grounds, against 216 gold claims.

## Decision thresholds

`scripts/sample_decisions.py` drew 16 outputs per validation example at temperature 1.0. An example's score for a statute is the share of its samples that include it, and `scripts/tune_thresholds.py` fits one threshold per statute to maximise macro-F1. To avoid scoring on the same examples it was fitted on, it also fits on one half of the applications and scores on the other, both ways round.

| Validation | Macro-F1 | 95% interval | Against greedy | No rejection | 102 | 103 | 112 | 101 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Greedy | 0.455 | 0.406–0.496 | | 0.655 | 0.573 | 0.564 | 0.436 | 0.048 |
| Majority vote | 0.470 | 0.423–0.514 | +0.015 (-0.017 to +0.049) | 0.645 | 0.605 | 0.606 | 0.460 | 0.033 |
| Thresholds, cross-validated | **0.546** | 0.496–0.591 | **+0.091** (+0.056 to +0.125) | 0.647 | 0.649 | 0.603 | 0.469 | 0.362 |
| Thresholds, fitted on all | 0.550 | 0.501–0.595 | +0.095 (+0.062 to +0.127) | 0.649 | 0.658 | 0.610 | 0.473 | 0.362 |

The fitted thresholds are 102: 0.25, 103: 0.44, 112: 0.44 and 101: 0.125. A 101 ground is predicted when 2 of 16 samples include it. So the adapter does give 101 real probability. It just isn't the top choice.

Two caveats:

- **The decision comes from the samples, the mapping and explanation from the greedy output.** A thresholded 101 can sit next to an explanation that never mentions it. The comparison will report thresholds as a separate decoding rule, alongside greedy, not instead of it.
- **Sampling costs 16 generations per example.** Each sample is short, but the cost still has to be counted against any gain.

## Failure review

These are 50 validation failures drawn with `scripts/sample_failures.py` (salt `failures-v1`, sampled within each gold class in proportion to its share of the 792 failures). Each one was read against the claim, its parents, the examiner's text and the model's output. The labels are in [gpu_sft_val_review.csv](../audits/failures/gpu_sft_val_review.csv), and the sampled records are next to it.

| Category | Count | What happens |
| --- | --- | --- |
| 102 and 103 confused | 15 | Right reference and often the right paragraph, wrong statute. 10 of the 15 are a 102 called a 103. |
| 112 or 101 missed | 14 | The art part is right, or there is none, but the 112 or 101 ground is missed. |
| Both of the above | 6 | A missed 112 or 101 together with a 102/103 swap. |
| Allowable claim rejected | 7 | All seven are claims with no examiner reasons for allowance, so the rejection can't be checked against the examiner's view. In three, the mapped passage plausibly discloses the limitation. |
| Rejected claim allowed | 5 | Four are empty outputs: no mapping and no explanation. One writes reasons for allowance while marking a limitation disclosed. |
| 112-only claim rejected on art | 2 | One blanket 112 over claims 1-21. One possible label error: the examiner's 112 wording differs from the claim text. |
| Truncated | 1 | The explanation loops on a list of citations. |

**102 against 103.** When the model wrongly says 103, it nearly always cites only one of the references it names. That holds for all 153 such errors on validation. But the same is true of 286 of the 287 103 predictions on claims that really have a 103 ground, because a dependent claim's own limitations usually rest on one reference. So the decision can't be read off the mapping: a rule that turns single-reference 103s into 102s drops macro-F1 to 0.353. What decides 102 against 103 for a dependent claim is how its parent was rejected, which the input doesn't show. Swaps hit 32% of dependent 102/103 claims against 23% of independent ones.

**112 and 101 are more learnable than the baseline review suggested.** 21 of the reviewed failures miss a 112 or 101 ground. 10 of those grounds are visible in the claim or its parents: a "signal" or "computer readable medium" claim, or antecedent basis such as "the compartment". 11 are inherited from a claim the model only sees as a parent, or not at all.

**Invented reference names.** Explanations name references ("Carpenter", "the combination of Jouppi and Callan") that the input never names: it carries document IDs and titles only. The examiner targets name references by inventor, so the model learned to write names it can't know. The scorer checks IDs, not names, so this doesn't show up in citation precision. It is still a faithfulness defect.

**Truncation is all citation loops.** All 30 unreadable outputs (1.9%) run to the token limit inside a citation list: paragraph numbers, figure numbers or reference numerals counting upward. 23 of them are art rejections. A cap on explanation length, or a repetition stop in the grammar, would turn these into readable, if long, outputs.

**Always "supported".** Partially supported art rejections were filtered out of training, so every art-rejection target says "supported", and the adapter now says so every time. The baseline marked 12% of its art rejections "unsupported". The DPO pairs that claim full support for a partially supported rejection are what should restore that signal.

## What this means for the next steps

- **The ablation.** The cheapest single change to test is cutting the citation loops, either by capping explanation length in the grammar or by dropping training targets with very long citation lists. That fixes the 1.9% unreadable outputs without touching the decision. The other candidate is training on 101 and 112 targets with more weight, which addresses the largest per-class gap directly.
- **Report thresholds as a decoding rule.** They are fitted on validation only and applied unchanged to test and OOD.
- **DPO.** Pairs should target the two defects SFT introduced or kept: claiming support that isn't there, and inventing reference names. The 102-against-103 confusion is mostly about the parent claim, so it may stay a ceiling for every variant.
