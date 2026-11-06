# Dataset v1: audit and manifest

**What this covers.** Two manual audits of 170 examples each, the fixes they led to, and dataset v1: the full 1% sample built with those fixes, and its manifest.

**Result.**

- **Decision labels hold up.** In a random sample of the full build, 3 of 90 evaluation decisions were wrong (3.3%, 95% interval 1–9%). Two came from parser gaps, now fixed. One is an examiner's typo.
- **Mappings and explanations are the weak part.** 36 of 50 audited evaluation mappings were fully right. After the fixes, an estimated 8% of evaluation examples (95% interval 4–15%) still have a wrong mapping or explanation target.
- **Audited errors are out of evaluation.** All audited errors left in v1 are in train.
- **v1 size:** 20,939 train, 1,597 validation, 1,627 test and 1,194 OOD examples, plus 34,735 + 2,750 preference pairs. Every spec size target is met or exceeded.

Raw files:

- First audit: [verdicts](dataset_audit_verdicts.jsonl), [sample](dataset_audit_sample.json) with each target's hash, [summary](dataset_audit_summary.json) with where each error went in v1
- Second audit: [verdicts](dataset_audit_v2_verdicts.jsonl), [sample](dataset_audit_v2_sample.json), [summary](dataset_audit_v2_summary.json) with error rates
- [dataset_v1_manifest.json](dataset_v1_manifest.json): the manifest

Pipelines:

- `scripts/audit_dataset.py` ([configs/dataset_audit.yaml](../../configs/dataset_audit.yaml), [configs/dataset_audit_v2.yaml](../../configs/dataset_audit_v2.yaml))
- `scripts/build_manifest.py` ([configs/dataset.yaml](../../configs/dataset.yaml))

## The first audit

### How it was done

- **Sample.** 80 train, 30 validation, 40 test and 20 OOD examples. Within each split, examples were taken round-robin over decision classes in salted-hash order (`audit-v1`), so rare classes such as 101+112 are represented. The sample was drawn from the 500-action build, before the fixes below.
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

### Findings

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

### Fixes

| Fix | Effect |
| --- | --- |
| Rejection statements may omit the verb, carry a short aside, or say "in the alternative" | 63 more statements found. OARD disagreement in the SFT corpus halves (41 → 20 examples) |
| "Diehr … (1981)" is no longer read as a paragraph cite | Fewer spurious cited passages |
| A figure-only cite ends a mapping window | A limitation supported by a figure isn't credited to the next paragraph cite |
| A mapped passage must contain at least 10% of the limitation's content words | Removes 17 of 23 audited wrong limitation mappings, and 13 of 77 correct ones |
| "Claims 1-4, 7 and 12-13 have been withdrawn" is read as withdrawn | Withdrawn claims no longer get a rejection target |
| Evaluation examples with an audited error are dropped (not moved to train) | 6 examples in v1, whose errors (examiner typos, a cite borrowed from another claim, group explanations) no rule catches |

Where each of the 46 examples with an error ended up in v1:
- **Dropped (35).** Mostly they now have rejection grounds whose evidence is cited only by column and line, or a partial mapping that keeps them out of evaluation.
- **Target changed (4).**
- **Unchanged (7).** All 7 are in train.

## The second audit

The first audit sampled the 500-action build, about 25 times smaller than the full one, and over-sampled rare classes on purpose, so it couldn't give error rates. The second audit was run on the full build before the fixes below.

**Method.** A simple random sample within each split (salt `audit-v2`): 80 train, 30 validation, 40 test and 20 OOD, with no example from the first audit. The sheets, the rubric and the four independent model reviewers were the same as in the first audit. The brief added two points from the first audit's adjudication: double patenting isn't a missing 101 ground, and a target that follows an examiner's mistyped reference isn't a pipeline error.

I checked the decision errors and a few of the others against the source text, and overruled none. The other verdicts stand as the reviewers gave them.

| | Evaluation (90) | Validation (30) | Test (40) | OOD (20) | Train (80) |
| --- | --- | --- | --- | --- | --- |
| Any error | 9 (10%; 5–18%) | 3 | 6 | 0 | 6 (7.5%; 4–15%) |
| Wrong decision | 3 (3.3%; 1–9%) | 1 | 2 | 0 | 3 (3.8%; 1–11%) |
| Wrong mapping | 3 (3.3%; 1–9%) | 1 | 2 | 0 | 1 (1.3%; 0–7%) |
| Off-claim or wrong explanation | 6 (6.7%; 3–14%) | 2 | 4 | 0 | 3 (3.8%; 1–11%) |
| Label leaked into input | 0 | 0 | 0 | 0 | 0 |

Ranges are 95% Wilson intervals. Of the audited 102/103 targets, 36 of 50 evaluation mappings and 27 of 60 train mappings were fully right. Train's rate is lower because it keeps partial mappings; 18 of its 32 "partly" verdicts are empty mappings.

**A reviewer's leak concern.** A cited reference with few visible passages can point at the cited one. The pool of examiner-cited passages is the same for every claim in an action, so this doesn't reveal a per-claim decision. It's a known property of the `action_cited` condition, and the `retrieved` condition doesn't have it.

### Fixes

| Fix | Effect |
| --- | --- |
| Every rejection statement in a text block is read, not only the first | In a sample of 600 actions, 33 had lost at least one statement, about 3% of all statements. A claim could lose a whole ground, as HV850D45:6 lost its 103 |
| Paragraph cite lists keep every item ("para. 31, 32, 43-45, and 78"; "[0050, 0055]"), and all items support the same limitation | Together with the fix above, 676 more examples have a mapping (16,258, from 15,582) |
| Statements may carry a parenthetical aside ("(as best understood)") or the typo "area rejected" | Two audited examples had lost a ground this way |
| Evaluation examples with an audited error are dropped | 7 examples |

Two earlier bugs surfaced while checking the full build, before this audit's sample was drawn:

| Fix | Effect |
| --- | --- |
| A claim with no limitations of its own ("An apparatus comprising a sensor according to claim 1") is flagged `mapping_partial` when rejected over art | 46 evaluation examples with no mapping had passed as fully grounded. They're out of evaluation |
| A claim listed only as "4. (canceled)" is excluded | 234 records, 60 of which had reached the splits with a label and no claim text |

The fixes moved numbers as expected. OARD disagreement falls from 1,218 to 1,018 examples, and the 93 mid-block statements in a 1,500-action sample all turned out to be real rejections.

**Where the audited examples ended up.**
- All 9 evaluation errors are gone: 7 dropped by verdict, 2 no longer pass the evidence rules once the fix gave them their missing grounds.
- 10 other audited evaluation targets changed, mostly from fuller mappings, and weren't reviewed again.
- In train, 3 examples with errors are gone, 1 has a changed target and 3 are unchanged.

### What remains

**Estimated error in v1 evaluation.** This takes the 90 audited evaluation examples and removes the 2 errors the fixes address, so it assumes the fixes work as they did on those examples. No sample of the fixed build has been audited.
- **Decision labels: about 1%** (1 of 90; 95% interval 0–6%). The remaining case is an examiner's "5-19" where 15-19 is meant.
- **Mappings or explanations: about 8%** (7 of 90; 95% interval 4–15%). These are group explanations covering other claims, and passages that follow the examiner's pin but don't disclose the limitation. No rule catches these, so scores on mapping and explanation need this caveat.

**Not fixed.**
- **Empty mappings in train.** 4,550 of train's 14,912 102/103 examples (31%) have no mapping, down from 33%. They're flagged `mapping_partial` and kept out of evaluation, but an SFT model trained on them could learn to leave mappings empty. Whether to filter them is a decision for the training config.
- **Garbled reference lists.** "over Best and Suzuki in view Naidenov of et al." resolves only the numbered reference. 1 of 170 audited.
- **Withdrawn and rejected claims.** An action can both withdraw a claim and reject it. The target follows the rejection. 1 of 170 audited, marked unclear.

**Automatic checks.** They run on every example:
- every mapped limitation appears verbatim in its claim
- every mapped passage shares words with its limitation
- no two examples share an input with different decisions
- provenance is complete
- every evaluation example passes all of the checks

Train has a few exceptions out of 20,939. 61 examples from 14 applications name a parent claim missing from the input, because the listing and the text disagree or the parent isn't in the listing. 30 have no explanation or examiner text.

## Dataset v1

| | Train | Validation | Test | OOD test |
| --- | --- | --- | --- | --- |
| Examples | 20,939 | 1,597 | 1,627 | 1,194 |
| Spec target | about 20,000 | 1,000 | 1,000 | 500–1,000 |
| Families | 3,264 | 331 | 322 | 323 |
| 102 / 103 / 112 / 101 | 6,544 / 8,554 / 3,231 / 1,591 | 564 / 476 / 309 / 118 | 401 / 549 / 213 / 95 | 126 / 218 / 690 / 18 |
| No rejection | 3,843 (18.4%) | 362 (22.7%) | 518 (31.8%) | 232 (19.4%) |
| Preference pairs | 34,735 | 2,750 | | |

**Source.** All 10,588 eligible first actions in the 1% sample (the salted-hash order means the earlier 500 are a prefix). They give 156,165 claim records and 28,541 examples from 4,647 applications, an 18% yield. The fetch took about 20 hours and 92,034 API requests. Run it with `scripts/build_dataset.sh`; it resumes from the cache.

What to keep in mind when using these numbers:

- **The biggest exclusion remains column and line cites.** 62,006 records rejected over art have evidence the input can't show, because the full-text XML has no column or line numbers. Together with records with no prior art text at all (28,071), this means rejections over granted patents are underrepresented compared with those over published applications.
- **Class balance differs between train and evaluation.** No-rejection is 18% of train but 32% of test, because a no-rejection target is more likely to pass the fully grounded rule. Report per-class scores next to any aggregate, and don't read train's balance into test.
- **OOD is mostly 112.** 112 is the largest class in technology centers 1600 and 1700, and 101 has only 18 examples there, so OOD 101 scores will be noisy.
- **Family links barely apply.** Only 9 families span more than one application, which is expected from a 1% random sample. The disjointness assert still runs.
- **Prior art overlap is measured, not removed.** 53 validation and 69 test applications share a cited document with train; OOD shares none.
- **Preference pairs far exceed the spec's 3,000.** The DPO phase should subsample them, with the mix decided then. The pairs are still template corruptions. Harder negatives need the fine-tuned model.

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

**Next.** The corpus is final for the baselines and the SFT phase. Fixing the remaining mapping and explanation errors would need more than rules; a third audit is only worth running if the pipeline changes again.
