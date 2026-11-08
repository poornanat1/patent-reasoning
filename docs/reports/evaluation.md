# Evaluation harness

**What this covers.** How every variant is scored: the decision metrics, the citation checks, the slices, and the statistics. The rubric is fixed before any variant is trained, so the comparison can't be tuned to a result.

Code:
- `src/patent_reasoning/eval/decision.py`: decision metrics and bootstrap
- `src/patent_reasoning/eval/faithfulness.py`: citation checks
- `scripts/score.py`: runs both on one variant's generations for one split

```bash
uv run python scripts/score.py configs/generation/base_few_shot.yaml --split test \
    --compare configs/generation/base_zero_shot.yaml
```

Scores go to `<out_dir>/<split>.scores.json` and MLflow.

## Decision

Five classes: no rejection, 102, 103, 112 and 101. A claim can have several grounds, so each class is scored as its own yes/no question. For example, a claim rejected under 103 and 112 that the model rejects under 103 alone is a true positive for 103 and a false negative for 112. "No rejection" is the class of an empty set of grounds.

| Metric | Definition |
| --- | --- |
| Per-class precision, recall, F1 | Over all examples in the split |
| Macro-F1 | Mean of the per-class F1 over the classes present in the split's gold labels (the headline metric) |
| Exact match | The predicted set of statutes equals the gold set |
| False-rejection rate | Share of claims the examiner found allowable that the model rejects |
| False-allowance rate | Share of rejected claims the model finds allowable |
| Unreadable rate | Outputs that aren't JSON, such as ones truncated at the token limit |

**Invalid outputs.** The decision is read from the grounds even when the output breaks the evidence-status rule, because the grammar guarantees the grounds are well-formed. Validity is reported on its own (`validity.valid_rate`). An unreadable output predicts nothing: it is a false negative for every gold class and a false positive for none.

## Citations

Computed from each output's claim mapping and grounds. Unreadable outputs contribute nothing here.

| Metric | Definition |
| --- | --- |
| Citation existence | Share of cited passages that are in the input and belong to the document they are cited for |
| Invalid reference rate | Share of ground references that aren't documents in the input |
| Citation precision | Share of cited passages that exist and contain at least 10% of the limitation's content words |
| Unsupported-assertion rate | Share of limitations marked disclosed with no valid passage |
| Art rejection without citation | Share of predicted 102/103 rejections that cite no valid passage at all |
| Support overclaim rate | Share of "supported" evidence statuses where some limitation isn't marked disclosed with a valid passage |
| Examiner pin recall | Of the limitations the reference mapping ties to passages (the examiner's resolved pin cites), the share the model marks disclosed citing at least one of those passages |
| Examiner pin precision | For those limitations, the share of the model's cited passages that the examiner also cited |

**What these measure and what they don't.**
- **Lexical support is a proxy.** The 10% floor is the one the reference mappings use. It catches a passage about something else. It doesn't catch a passage that uses the same words to say something different, which only a reader can judge. The manual failure review samples these.
- **Generation doesn't constrain IDs.** Citation existence is a real measurement, not 100% by construction.
- **Pin agreement is one examiner's view.** A passage the examiner didn't cite can still disclose the limitation, so pin precision understates correct citations. It is reported because it is the only independent reference available.

## Slices and out-of-distribution sets

- **OOD by technology.** The OOD test set is every family examined in technology centers 1600 (biotechnology) and 1700 (chemistry), none of which appears in train ([dataset_v1.md](dataset_v1.md)).
- **Technology center and mail year.** Every score also breaks the decision metrics down by technology center and by the action's mail year. Slices under 30 examples list only their size.
- **No temporal holdout.** Train, validation and test all span 2009–2014, so a period-based OOD set would need a new split with train restricted to earlier years. With five usable years, a holdout of 2014 would remove 26% of the data. The year slices show whether scores drift by period instead, and a temporal split can be added if they do.

## Statistics

- **Bootstrap by application.** Macro-F1 has a 95% interval from 1,000 bootstrap resamples. Each resample draws whole applications, not claims, because claims in one office action share prior art and an examiner.
- **Paired comparisons.** `--compare` runs a paired bootstrap on the examples both variants generated: the macro-F1 difference, its 95% interval, and the share of resamples in which the variant beats the baseline.
- **Seeded.** Bootstrap seeds are fixed, so a rescore gives the same intervals.

## Cost

Each score records generation seconds, prompt tokens and completion tokens. Training cost is added for the trained variants.
