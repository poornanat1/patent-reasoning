# Baselines: frozen Qwen3-8B

**What this covers.** The two untrained baselines every later variant is compared against, and a manual review of 50 of their failures. Both use the base model with deterministic, schema-constrained generation ([base_model.md](base_model.md)) and are scored with the harness in [evaluation.md](evaluation.md).

| Variant | Config | Prompt |
| --- | --- | --- |
| Zero-shot | `configs/generation/base_zero_shot.yaml` | System prompt and the example |
| Few-shot | `configs/generation/base_few_shot.yaml` | Plus five demonstrations from train: one each for 102, 103, 112, 101 and no rejection |

## Reference scores: GPU baselines

These are the scores every trained variant is compared against. They come from the setup the comparison uses: one A100 80 GB, vLLM, Qwen3-8B in bf16 at revision `b968826`, and the final grammar ([base_model.md](base_model.md#update-the-cuda-backend)). Configs: `configs/generation/gpu_base_zero_shot.yaml` and `gpu_base_few_shot.yaml`.

| | Zero-shot val | Zero-shot test | Zero-shot OOD | Few-shot val | Few-shot test | Few-shot OOD |
| --- | --- | --- | --- | --- | --- | --- |
| Macro-F1 | 0.241 | 0.276 | 0.162 | 0.248 | 0.258 | 0.147 |
| 95% interval | 0.216–0.263 | 0.253–0.298 | 0.133–0.186 | 0.222–0.271 | 0.233–0.281 | 0.120–0.171 |
| Exact match | 0.320 | 0.403 | 0.163 | 0.354 | 0.384 | 0.172 |
| False-rejection rate | 0.503 | 0.438 | 0.526 | 0.373 | 0.376 | 0.435 |
| False-allowance rate | 0.265 | 0.221 | 0.489 | 0.317 | 0.293 | 0.531 |
| Valid outputs | 100.0% | 99.9% | 99.4% | 99.8% | 99.9% | 99.4% |
| Citation precision | 0.910 | 0.893 | 0.855 | 0.914 | 0.906 | 0.893 |
| Art rejections citing nothing | 0.008 | 0.018 | 0.067 | 0.007 | 0.009 | 0.029 |
| Examiner pin recall | 0.373 | 0.402 | 0.385 | 0.321 | 0.344 | 0.345 |
| Seconds per example | 0.35 | 0.35 | 0.34 | 0.50 | 0.50 | 0.47 |

**Per-class F1** (predictions in brackets):

| Class | Zero-shot val | Zero-shot test | Zero-shot OOD | Few-shot val | Few-shot test | Few-shot OOD |
| --- | --- | --- | --- | --- | --- | --- |
| No rejection | 0.414 (507) | 0.551 (535) | 0.262 (576) | 0.458 (616) | 0.553 (647) | 0.294 (639) |
| 102 | 0.384 (384) | 0.385 (400) | 0.293 (297) | 0.520 (721) | 0.436 (690) | 0.319 (419) |
| 103 | 0.406 (706) | 0.445 (688) | 0.256 (313) | 0.244 (253) | 0.268 (286) | 0.080 (106) |
| 112 | 0.000 (1) | 0.000 (2) | 0.000 (1) | 0.000 (0) | 0.009 (1) | 0.040 (16) |
| 101 | 0.000 (1) | 0.000 (2) | 0.000 (1) | 0.017 (3) | 0.021 (2) | 0.000 (7) |

**Few-shot against zero-shot (paired bootstrap of macro-F1):**

| Split | Difference | 95% interval | Few-shot ahead in |
| --- | --- | --- | --- |
| val | +0.007 | -0.015 to +0.028 | 73% |
| test | -0.018 | -0.040 to +0.006 | 6% |
| OOD | -0.016 | -0.037 to +0.010 | 11% |

**What changed from the laptop runs below:**

- **Few-shot cites again.** On the laptop, few-shot left 98% of its prior-art rejections citing nothing. On the GPU it's under 3% on every split. The laptop demonstrations were serialized with spaces the model never writes under the grammar, and that mismatch is now fixed, so the earlier failure was most likely a format artifact rather than a property of few-shot prompting.
- **112 and 101 almost vanish.** In bf16 the model predicts 112 or 101 at most 16 times on any split, against 213–690 gold 112 claims, so both classes score near zero. Macro-F1 averages over all five classes, which holds the baselines to 0.24–0.28 on validation and test even though the best of the other three classes reaches 0.41–0.55.
- **The two baselines are still a tie.** No paired difference is significant.

The rest of this report covers the laptop (MLX 4-bit) runs and the failure review. Those numbers are a development record, kept because the failure categories still apply.

## Laptop scores (MLX 4-bit)

| | Zero-shot val | Few-shot val | Zero-shot test | Few-shot test | Zero-shot OOD | Few-shot OOD |
| --- | --- | --- | --- | --- | --- | --- |
| Examples | 1,597 | 1,597 | 1,627 | 1,627 | 1,194 | 1,194 |
| Macro-F1 | 0.215 | 0.237 | 0.239 | 0.221 | 0.156 | 0.156 |
| 95% interval | 0.190–0.240 | 0.205–0.267 | 0.216–0.262 | 0.196–0.246 | 0.130–0.181 | 0.130–0.181 |
| Exact match | 0.287 | 0.199 | 0.371 | 0.271 | 0.149 | 0.137 |
| False-rejection rate | 0.536 | 0.285 | 0.523 | 0.293 | 0.608 | 0.358 |
| False-allowance rate | 0.270 | 0.624 | 0.236 | 0.623 | 0.498 | 0.628 |
| Valid outputs | 99.2% | 89.0% | 99.3% | 90.7% | 97.6% | 93.2% |
| Citation existence | 0.982 | 1.000 | 0.984 | 1.000 | 0.949 | 0.951 |
| Citation precision | 0.914 | 0.962 | 0.903 | 0.952 | 0.846 | 0.927 |
| Art rejections citing nothing | 0.132 | 0.978 | 0.145 | 0.962 | 0.182 | 0.949 |
| Examiner pin recall | 0.322 | 0.009 | 0.362 | 0.016 | 0.358 | 0.016 |
| Seconds per example | 8.7 | 8.4 | 10.1 | 6.9 | 6.3 | 5.1 |
| Prompt tokens | 4.98M | 14.35M | 5.18M | 14.72M | 3.48M | 10.48M |

**Few-shot against zero-shot (paired bootstrap of macro-F1):**

| Split | Difference | 95% interval | Few-shot ahead in |
| --- | --- | --- | --- |
| Validation | +0.022 | −0.015 to +0.057 | 89% of resamples |
| Test | −0.018 | −0.047 to +0.014 | 11% |
| OOD | +0.001 | −0.027 to +0.028 | 51% |

No difference is significant. The two baselines score about the same; they fail in different ways.

**Per-class F1:**

| Class | Gold (test) | Zero-shot test | Few-shot test | Gold (OOD) | Zero-shot OOD | Few-shot OOD |
| --- | --- | --- | --- | --- | --- | --- |
| No rejection | 518 | 0.480 | 0.465 | 232 | 0.218 | 0.303 |
| 102 | 401 | 0.168 | 0.123 | 126 | 0.205 | 0.096 |
| 103 | 549 | 0.504 | 0.320 | 218 | 0.341 | 0.188 |
| 112 | 213 | 0.025 | 0.078 | 690 | 0.014 | 0.162 |
| 101 | 95 | 0.019 | 0.120 | 18 | 0.000 | 0.034 |

Validation per-class scores follow the same pattern as test and are in the score files.

Seconds per example are wall-clock on an M5 Pro and include stretches on battery, so they overstate the cost of a run on mains power.

## What the baselines do

**Zero-shot answers almost everything with 103.** It predicts 103 for 936 validation claims against 476 gold, and calls 311 of the 453 claims the examiner rejected under 102 alone a 103 (69%). It almost never raises 112 or 101: of the 403 validation claims with either ground, it finds the ground in 9. Its citations are mostly real: 98% of cited passages exist in the input, and 91% share words with their limitation.

**Few-shot stops citing.** The demonstrations moved the model towards "no rejection" (1,030 predictions against 362 gold) and raised the 112 and 101 scores, but every one of its 1,030 allowances, and 535 of its 547 prior-art rejections, comes with an empty claim mapping. The model asserts a limitation isn't disclosed without looking for a passage. All 175 invalid outputs are prior-art rejections whose evidence status isn't allowed for one.

The likely cause is the demonstration set: three of the five (112, 101, no rejection) show an empty mapping or a "not applicable" status, and the model copies that shape even when it rejects on prior art. This is inferred from the counts, not tested. The demonstrations were left unchanged for the test and OOD runs, so few-shot is reported as a weak baseline rather than tuned on validation.

## Out of distribution

The OOD set (technology centers 1600 and 1700, biotechnology and chemistry) is harder for both baselines: macro-F1 falls from about 0.23 on test to 0.156, and the intervals don't overlap. The main reason is the label mix. 112 is the most common ground in OOD, on 690 of 1,194 claims (58%), against 13% in test. Both baselines are weakest on exactly that class (F1 0.014 and 0.162). Biotechnology (1600) is the hardest center, at 0.080 for zero-shot and 0.110 for few-shot; chemistry (1700) is 0.184 and 0.169.

Citation behaviour holds up: zero-shot's citation existence falls only from 0.98 to 0.95, and its pin recall is unchanged at 0.36. What drops is the decision, not the ability to find passages.

## Failure review

50 validation failures, drawn with `scripts/sample_failures.py` (salt `failures-v1`, sampled within each gold class in proportion to its share of failures): 40 from zero-shot and 10 from few-shot. A failure is an example whose predicted set of statutes differs from the gold set. Each was read against the claim, its parents, the examiner's text and the model's output. The labels are in [failures/base_val_review.csv](../audits/failures/base_val_review.csv) and the sampled records next to it.

**Zero-shot (40):**

| Category | Count | What happens |
| --- | --- | --- |
| 102 and 103 confused | 15 | The model finds the disclosure the examiner relied on, often the same paragraph, but names the wrong statute. 12 of the 15 are a 102 called a 103. |
| 112 or 101 missed | 15 | The ground usually comes from somewhere the model can't see in the claim: 112 inherited from a parent or another claim (antecedent basis, an indefinite term in claim 1), or a blanket rejection of a claim range. The misses that are visible in the claim itself are a computer-readable medium, a signal per se, and a method not tied to a machine (101). |
| Allowable claim rejected | 6 | Two reject under 103 while the explanation says the limitations aren't disclosed. Three have no recorded reasons for allowance, so the rejection can't be checked against the examiner's view. |
| Prior art missed | 3 | One rests on official notice, which isn't in the input. One misses the passage the examiner cited. One may be a label error: the examiner's text says "20 mol%" and the claim "80 mol%". |
| Reasoning error | 1 | Treats 0.5 ohm-cm as less than 10 mohm-cm. |

**Few-shot (10):** 9 predict no rejection with an empty mapping, including four where the examiner cites the disclosing paragraph. The tenth gets 101 right, invents a 112 and drops the 102.

**Possible label errors.** Three of the 40 zero-shot failures may be wrong in the data rather than in the model: a 112 statement about claim 6 attached to claim 12, a "rejected as dependent upon a rejected base claim" (an objection) labelled 112, and the mol% mismatch. That's in line with the dataset audit's estimated residual of about 8% for mappings and explanations ([dataset_v1.md](dataset_v1.md)). None is corrected here, because the test and OOD splits must stay as they were frozen.

**Self-contradiction.** Across validation, 135 of zero-shot's 1,095 prior-art rejections (12%) carry the status "unsupported", meaning the model rejects while saying the art doesn't support the rejection. Few-shot does this on 281 of 547 (51%).

## What this means for training

- **102 against 103 is the cheapest gain.** The model already finds the right passage; it needs to learn that one reference disclosing every limitation is a 102. SFT on targets with full mappings should fix most of these.
- **112 and 101 depend on context the input lacks.** Many 112 grounds are inherited from claims the model sees only as parents, or from claims it doesn't see at all. Training can teach the model to propagate a parent's defect, but grounds about claims outside the input can't be learned from it. These examples bound how far 112 recall can rise.
- **Rejecting without support should be penalised.** The 103-with-"unsupported" pattern is a natural rejected side for DPO preference pairs, and a natural failure for a GEPA metric to score.
- **Few-shot needs demonstrations with full mappings.** Before few-shot is used inside DSPy, the demonstration set should include prior-art rejections with complete mappings, and be checked on validation.
