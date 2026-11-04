# Splits and preference pairs

> **Superseded numbers.** Parser fixes from the dataset audit changed these counts; the current ones are in [dataset_v1.md](dataset_v1.md).

**What this covers.** The SFT corpus (1,144 examples) split into train, validation, in-distribution test and out-of-distribution test sets, with no application family in more than one. Then a first set of DPO preference pairs built from the train and validation splits.

**Result.**

- **Splits:** 751 train, 99 validation, 129 test and 33 OOD test examples. No family or application crosses splits, and no training input cites an evaluation application's own publication.
- **Preference pairs:** 1,344 for training and 184 for validation.

The evaluation sets are far below the spec's targets (1,000 validation, 1,000 test, 500–1,000 OOD). They grow with the corpus.

Raw numbers: [splits_summary.json](splits_summary.json), [preferences_summary.json](preferences_summary.json). Pipelines:

- `scripts/build_splits.py` ([configs/splits.yaml](../../configs/splits.yaml))
- `scripts/build_preferences.py` ([configs/preferences.yaml](../../configs/preferences.yaml))

## Families

Two applications share a family when continuity or a shared foreign priority filing links them, directly or through other applications. Continuity covers continuations, divisionals, continuations-in-part and national-stage entries of the same PCT application. The data comes from the Open Data Portal's `continuity` and `foreign-priority` endpoints.

- **Links:** 177 of the 206 applications have at least one.
- **Shared families:** none of the 206 share a family with another. That's expected from a 1% random sample of applications. The check matters more as the sample grows, and the pipeline asserts it.

## Split rules

| Split | How assigned |
| --- | --- |
| OOD test | Every family examined in technology centers 1600 (biotechnology) or 1700 (chemistry): 8% of the corpus, and a real shift from the electrical and mechanical bulk |
| Validation / test / train | The other families, by salted hash of the family ID (`splits-v1`): 10% / 10% / 80% |

**Why not hold out by period.** The latest mailing year (2014) is 26% of the corpus, too much to hold out at this size. The technology holdout fits the spec's OOD size.

**Fully grounded only.** Validation and both test sets keep only fully grounded examples:
- claim text from the publication, not OCR
- no `mapping_partial` or `restatement_low_overlap` flag
- an explanation for every rejection

132 examples of evaluation families fail this and are dropped, not moved to train, which would leak the family. The OOD set loses the most (58 of its 91 examples).

## Splits

| | Train | Validation | Test | OOD test |
| --- | --- | --- | --- | --- |
| Examples | 751 | 99 | 129 | 33 |
| Applications (= families) | 137 | 15 | 20 | 12 |
| 102 | 217 | 21 | 23 | 5 |
| 103 | 357 | 29 | 59 | 9 |
| 112 | 81 | 10 | 9 | 20 |
| 101 | 49 | 10 | 8 | 0 |
| No rejection | 132 (17.6%) | 45 (45.5%) | 39 (30.2%) | 3 (9.1%) |
| Evidence status supported / partial / n.a. | 422 / 151 / 178 | 50 / 0 / 49 | 82 / 0 / 47 | 14 / 0 / 19 |

**Leakage checks:**
- No family or application spans two splits. The build asserts both.
- No train example cites an evaluation application's own publication. If one did, the passage would repeat the evaluation claim's text. The check found 0 such examples, so none were dropped.
- No evaluation example shares a cited document with any train example.

**Caveats.**
- With 15–20 families per evaluation split, class mix swings a lot between splits. Validation is 45% no-rejection against 18% in train, because a few applications contribute many allowable claims. Scores need per-class reporting and bootstrap intervals over families, not over claims.
- The OOD set is mostly 112 rejections. Chemical and biotech claims are often rejected as indefinite, and their art rejections more often cite non-patent literature, which the corpus excludes.

`data/processed/splits/manifest.json` lists every example ID and the SHA-256 of each split file. `families.json` records application → family → split.

## Preference pairs

Each pair holds the example's input, the target as the chosen output, and a rejected output. The rejected output is a programmatic corruption: a valid `ModelOutput` that cites only documents and passages present in the input. So the two outputs differ in substance, not format. Pairs come only from the train and validation splits, and the build refuses test splits.

| Type | What the rejected output does | Train | Val |
| --- | --- | --- | --- |
| `citation_corruption` | Cites a different passage of the input, one the examiner didn't cite for this claim | 471 | 50 |
| `unsupported_rejection` | Keeps the rejection and "supported", but cites no passages | 471 | 50 |
| `false_rejection` | Rejects an allowable claim under 102 over the main cited reference, with each limitation "disclosed" by its best-matching passage | 132 | 42 |
| `overconfident_support` | Presents a partially supported rejection as fully supported, with invented mappings for the unmapped limitations | 138 | 0 |
| `false_allowance` | Allows a rejected claim, with every limitation "not disclosed" | 132 | 42 |
| **Total** | | **1,344** | **184** |

- **Coverage:** 692 of 751 train examples contribute at least one pair.
- **Validation has no `overconfident_support` pairs,** because fully grounded examples are never partially supported.
- **False allowances are capped.** They're limited to the number of false rejections in each split, chosen by hash. Without the cap there were 560, which would teach the model to reject far more than to allow and shift its decision prior.

**Limits.** The rejected outputs are templates. They teach the model what to avoid, such as citing the wrong passage, asserting support with no citations, or overclaiming. They don't teach it what good reasoning sounds like; that comes from SFT on examiner text. Harder negatives, like plausible but wrong mappings sampled from the SFT model itself, belong in the DPO phase once there is a model to sample from. The spec's target is about 3,000 pairs, and this seed set is 1,344. It grows with the corpus.
