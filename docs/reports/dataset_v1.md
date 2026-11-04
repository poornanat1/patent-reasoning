# Dataset v1: audit and manifest

**What this covers.** A manual audit of 170 examples drawn from the splits, the fixes it led to, and the dataset v1 manifest built after them.

**Result.**

- **Decision labels hold up.** After adjudication, 155 of 170 audited decisions were correct, 13 wrong and 2 unclear. Every wrong one traced to two parser gaps, both now fixed.
- **Claim-to-passage mappings were the weak part.** Of the 113 audited 102/103 targets, 57 mappings were right, 31 partly right and 25 wrong.
- **Evaluation sets are clean of every audited error.** All errors left in v1 are in train.
- **v1 size:** 754 train, 78 validation, 99 test and 21 OOD examples, plus 1,302 + 135 preference pairs.

Raw files:

- [dataset_audit_verdicts.jsonl](dataset_audit_verdicts.jsonl): one verdict per audited example
- [dataset_audit_sample.json](dataset_audit_sample.json): the drawn sample, with each target's hash
- [dataset_audit_summary.json](dataset_audit_summary.json): tallies, plus where each error went in v1
- [dataset_v1_manifest.json](dataset_v1_manifest.json): the manifest

Pipelines:

- `scripts/audit_dataset.py` ([configs/dataset_audit.yaml](../../configs/dataset_audit.yaml))
- `scripts/build_manifest.py` ([configs/dataset.yaml](../../configs/dataset.yaml))

## How the audit was done

- **Sample.** 80 train, 30 validation, 40 test and 20 OOD examples. Within each split, examples were taken round-robin over decision classes in salted-hash order (`audit-v1`), so rare classes such as 101+112 are represented. The sample was drawn from the build before the fixes below.
- **Sheets.** Each example was shown with:
  - the claim and its parents, and the prior-art documents in the input
  - the target with every mapped passage's text
  - the examiner's text and pin cites
  - every sentence of the office action that disposes of claims and names this claim, directly or through a range. This is so the label can be checked against the action itself, not against the parser's reading of it.
- **Reviewers.** Four independent model reviewers each took a quarter of the sample. Each judged five fields against a fixed rubric:
  - decision
  - claim text
  - mapping
  - explanation
  - leakage of the label into the input
- **Adjudication.** I re-checked their reported errors against the office actions and overruled three decision verdicts. The reasons are recorded in the verdicts file.
  - Two blamed the pipeline for a reference that the examiner had in fact mistyped. The action reads "Park et al., US PG-Pub 2010/0086317", and that publication is about a printer.
  - One counted a provisional statutory double-patenting rejection as a missing 101 ground. Double patenting is outside the task's 101 (eligibility) class.

  Model reviewers are not attorneys. Their verdicts are best read as a careful second reading that a person can check line by line, which the files allow.

## Findings

| Field | Correct / ok | Partly | Wrong | Unclear | n/a |
| --- | --- | --- | --- | --- | --- |
| Decision | 155 | | 13 | 2 | |
| Claim text | 166 | | 0 | 4 | |
| Mapping (102/103 only) | 57 | 31 | 25 | | 57 |
| Explanation | 132 | | 16 (15 off-claim) | | 22 |
| Label leaked into input | 170 none | | 0 | | |

- **Wrong decisions** (13) came from two gaps in reading rejection statements:
  - **Statements without a verb.** "Claims 3, 8, 13 … rejected under 35 U.S.C. 103(a)". This is 60 of the 1,564 statements in the sample's actions (4%). The parser required "is" or "are", so it dropped the whole rejection.
  - **Statements with an aside.** "Claims 1-8 and 10, per above incorporated art rejections, are in the alternative, are also rejected under 102(e)", or "Claims 1-3, as best understood, are rejected".
- **Wrong mappings** (25) had four causes:
  - **A cite resolved to the wrong document.** The examiner misspelled a name ("Erye" for Eyre), so the nearest name before the cite was another reference. Or the examiner mistyped a publication number.
  - **A figure-only cite.** In "a hinge (Fig. 13) and a latch ([0071])", the hinge was credited to [0071].
  - **Part of a cited range in the input.** When only one paragraph of "[0041]-[0048]" made the input, the limitation was mapped to that paragraph whether or not it was the relevant one.
  - **Partial mappings with no rows.** 102/103 targets marked partially supported, where the examiner's cites for this claim didn't reach the input. The reviewers marked some of these wrong because a mapping exists in the action. These are flagged `mapping_partial` and already kept out of evaluation.
- **Off-claim explanations** (15) are the examiner's shared paragraph for a group of claims ("claims 1-12 …"), or claim 1 boilerplate inherited by a dependent claim.
- **What held up:**
  - claim ranges
  - allowed and allowable-dependent claims (all 24 no-rejection examples were right)
  - 101/112 statute sets
  - amended claim text
  - no example leaked its label into the input

## Fixes

| Fix | Effect |
| --- | --- |
| Rejection statements may omit the verb, carry a short aside, or say "in the alternative" | 63 more statements found. OARD disagreement in the SFT corpus halves (41 → 20 examples) |
| "Diehr … (1981)" is no longer read as a paragraph cite | Fewer spurious cited passages |
| A figure-only cite ends a mapping window | A limitation supported by a figure isn't credited to the next paragraph cite |
| A mapped passage must contain at least 10% of the limitation's content words | Removes 17 of 23 audited wrong limitation mappings, and 13 of 77 correct ones |
| "Claims 1-4, 7 and 12-13 have been withdrawn" is read as withdrawn | Withdrawn claims no longer get a rejection target |
| Evaluation examples with an audited error are dropped (not moved to train) | 7 examples, whose errors (examiner typos, a cite borrowed from another claim, group explanations) no rule catches |

Where each of the 46 examples with an error ended up in v1:
- **Dropped (35).** Mostly they now have rejection grounds whose evidence is cited only by column and line, or a partial mapping that keeps them out of evaluation.
- **Target changed (2).**
- **Unchanged (9).** All 9 are in train.

The automatic checks run on every example and find no remaining problems:
- every mapped limitation appears verbatim in its claim
- every dependent claim has its parents
- every mapped passage shares words with its limitation
- no two examples share an input with different decisions
- provenance is complete

The exceptions are one train example with no explanation and one train example with no examiner text.

**Residual error estimate.** 58 audited evaluation examples survive the systematic rules. 7 of them (12%) still had an error, and only the audit caught them. All 7 were mapping or explanation errors; none was a decision error. So:
- About 12% of the 147 unaudited evaluation examples may have a wrong mapping or explanation target.
- The decision label error rate in evaluation is likely below 5% (0 of 58 seen).
- Mapping and explanation scores need this caveat; decision scores less so.

## Dataset v1

| | Train | Validation | Test | OOD test |
| --- | --- | --- | --- | --- |
| Examples | 754 | 78 | 99 | 21 |
| Families | 139 | 13 | 17 | 11 |
| 102 / 103 / 112 / 101 | 218 / 366 / 79 / 47 | 14 / 17 / 6 / 5 | 12 / 48 / 0 / 0 | 4 / 1 / 15 / 0 |
| No rejection | 132 (17.5%) | 45 (57.7%) | 39 (39.4%) | 3 (14.3%) |
| Preference pairs | 1,302 | 135 | | |

- **Why the evaluation sets shrank** (from 99 / 129 / 33):
  - The 10% passage floor turns more mappings partial, and validation and test keep only fully grounded examples.
  - The verbless-statement fix gave some evaluation claims a 102/103 ground whose evidence isn't in the input.
  - Audited errors were dropped.
- **The test set has no 101 or 112 example.** Macro-F1 over five classes can't be computed on it until the corpus grows, so test scores must be reported per class.
- **Pair mix:** more `overconfident_support` pairs (232, from 138), because more examples are partially supported.

**Manifest** ([dataset_v1_manifest.json](dataset_v1_manifest.json), with a copy at `data/processed/manifest.json`). It records:
- the git commit and schema version
- every stage's config, and every output file with its SHA-256, size and line count, from the OARD sample through the audit
- every example ID per split, with its application and family
- the preference pairs by type
- the audit result

The build fails if a split file no longer matches the splits manifest, if a family spans splits, or if a preference pair's example isn't in its split.

**Provenance.** Every example names:
- its application, office action, mail date and art unit
- the source of the decision (rejection statement, allowable dependent, allowed)
- whether OARD agrees
- the claim text source and whether it was verified
- the evidence condition
- the examiner's verbatim text and pin cites
- quality flags

Examiner statements remain what the examiner asserted. The audit verdicts are kept separately, so asserted facts and independent checks never mix.

**Next.** The evaluation sets are small and, in test, missing two classes. They grow only with the corpus: fetching the rest of the 1% sample (about 11.5k first actions) is the main lever, and the audit should be repeated on a fresh sample once it is built.
