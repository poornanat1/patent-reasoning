# Office Action Research Dataset: audit findings

This audits the USPTO Office Action Research Dataset, 2017 release (ODP product `PTOFFACT`). The generated reports are [oard_audit.md](oard_audit.md) (full data) and [oard_audit_sample.md](oard_audit_sample.md) (sample). Both can be regenerated with `scripts/audit_oard.py`.

## Data obtained

| Table | Zip | Unzipped | Rows | Documented rows |
| --- | --- | --- | --- | --- |
| office_actions | 83 MB | 390 MB | 4,384,532 | 4,384,532 |
| rejections | 95 MB | 695 MB | 10,133,179 | 10,133,179 |
| citations | 472 MB | 2.46 GB | 58,862,278 | 58,862,278 |

The zip sizes match the portal's metadata byte for byte. Every table has exactly the documented columns.

**Sample** (1% of applications with an office action mailed 2008–2014, chosen by salted SHA-256 of `app_id`):

| Measure | Count |
| --- | --- |
| Cohort applications | 1,415,733 |
| Sampled applications | 14,247 |
| Office actions | 30,693 |
| Rejection rows | 69,823 |
| Citation rows | 381,364 |
| Citation rows linked to a specific rejection | 87,356 |

The sample keeps every office action for a sampled application, so it includes 4,293 actions mailed in 2015–2017.

## What's clean

- **Identifiers.** Every `app_id` is 8 digits and every `ifw_number` is unique. All rejection rows join to an office action with the same `app_id`.
- **Citation links.** All 12.3M citations linked to an action join to a matching rejection row on `ifw_number` + `action_type` + `action_subtype`. Blank subtypes must be matched as equal for this to hold.
- **Dates.** `mail_dt` is ISO `YYYY-MM-DD` and 100% of values parse.
- **Claim numbers.** `claim_numbers` is a comma-separated list such as `"1,2,3,4"`. 100% of values fit the parser, so every rejection maps to specific claim numbers.
- **Flags vs. rows.** Wherever a rejection row exists, the matching office-action flag is set (`rows_only = 0` for 101/102/103/112).

## Issues and their consequences

1. **Citations identify documents, not paragraphs.** Links reach only the (office action, rejection type) level. No claim limitation is tied to a passage. Paragraph-level evidence has to come from the office action text itself (Day 4), so the claim-mapping target depends on parsing that text.
2. **Only 102 and 103 rejections carry citations.** That matches the law: 101 and 112 rejections don't rest on prior art. Evidence-grounded generation therefore applies to 102/103. For 101/112, the task is classification plus reasoning from the examiner's text.
3. **Label imbalance.** 103 appears in 79% of office actions, 102 in 42%, 112 in 35%, 101 in 11% and double patenting in 11%. Labels are multi-label per action. Macro-F1 is the right headline metric, and the splits should be stratified.
4. **Some flags have no rejection rows.** In the full data, 44,587 actions flag 112, 20,691 flag 101 and 19,370 flag 102 without a matching rejection row. Rejection rows, not flags, should be the label source.
5. **Parsing-quality flags.** 10.5% of actions have no form paragraphs, 7.1% are missing headers, and 4.8% raise rejections that don't match their form paragraphs. Excluding `rejection_fp_mismatch = 1` from gold evaluation data should give more reliable labels.
6. **Not every `action_type` is a rejection.** It also includes `objected` (1.18M rows), `cancelled` (611k) and `allowed` (278k). These describe claim status, which is useful context and gives "no rejection" negatives. They must not be counted as rejection labels.
7. **Fine-grained 101 subtypes exist,** such as `Non-Statutory-alice`, `-mayo`, `-data` and `-software`. They could support a finer 101 analysis later.
8. **Date outliers.** 28 office actions are dated 2001-01-01, and real volume starts in April 2008. 2008 is thin (6,837 actions), while 2010–2014 carries most of the cohort.
9. **Citation IDs are US-only.** `parsed` is blank for 0.65% of citations, likely foreign or non-patent literature. The rest are 7-digit US patent numbers or 11-digit pre-grant publication numbers, which can be fetched from the USPTO full text.
10. **Claim versions may differ.** An office action rejects the claims pending at that time, which are often amended. The Patent Claims Research Dataset holds the claims as published. Matching claim text to a rejection needs a version check (Day 5–6), and final rejections (CTFR, 32% of actions) are the most likely to involve amended claims.

## Decisions for next steps

- Take labels from `rejections` rows (statutory types 101/102/103/112 and double patenting), and keep objected/cancelled/allowed as claim status. *Update after the feasibility join:* `claim_numbers` is incomplete in about 18% of 102/103 rejections when compared with the office action text. See [feasibility.md](feasibility.md).
- Apply evidence-grounded targets to 102/103 only, and treat 101/112 as classification plus examiner-reasoning tasks.
- Pull office action text for sampled 102/103 actions first (Day 4), since paragraph-level evidence depends on it.
- Revisit the cohort filter once claims are joined: filter by publication year instead of mail year.
