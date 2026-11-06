# Claim-level records and evidence

> **Superseded numbers.** Parser fixes from the dataset audit and the full 1% sample build changed these counts; the current ones are in [claim_records_summary.json](claim_records_summary.json) and [dataset_v1.md](dataset_v1.md).

**What this covers.** Each claim in an application's first non-final office action becomes one record. The record holds the examiner's decision on that claim, its text and parent claims, its limitations, and the examiner's reasoning and pin cites for it. Each record is then paired with the cited prior art as passages, chosen under the fixed budget that every experiment uses.

**Result.**

- **Records:** 7,005 claim records from 500 office actions in 469 applications.
- **Grounded records:** 1,561 (22%). These have a trustworthy label and claim text, cited art with full text, and for 102/103 rejections at least one examiner pin cite resolved to a paragraph.
- **Evidence coverage:** the primary evidence setting puts at least one of the examiner's own cited passages in front of the model for 91% of claims that have resolved pin cites.

Raw numbers: [claim_records_summary.json](claim_records_summary.json) and [evidence_summary.json](evidence_summary.json). Pipelines:

- `scripts/build_claim_records.py` ([configs/claim_records.yaml](../../configs/claim_records.yaml))
- `scripts/build_evidence.py` ([configs/evidence.yaml](../../configs/evidence.yaml))

## Sample

- **Cohort:** the first non-final action (CTNF) of each application, mailed 2008–2014, with a 102/103 rejection and clean OARD parsing flags. These come from the 1% application sample, which contains about 11,500 such actions. I chose first actions because the pending claims there are closest to the claims as filed.
- **Size:** 500 actions, the first 500 by salted hash (`claims-v1`). The plan was 1,000. The Open Data Portal began throttling halfway through the download, and fetching the second 500 would have taken hours. `n_office_actions` in the config sets the size, and a rerun picks up the same 500 plus the new ones.

| Stage | Count |
| --- | --- |
| Office actions | 500 |
| With office action text | 500 |
| With claim text (publication or as-filed OCR) | 490 |
| With at least one claim record | 469 |
| Claims the action decides (rejected, allowed or allowable; withdrawn claims excluded) | 7,196 |
| Of which claim text is missing | 191 |
| Claim records | 7,005 |

## Labels

Every claim's decision comes from the action's own text:
- **Rejections:** the rejection statements ("Claims 1-5 are rejected under 35 U.S.C. 103(a) as being unpatentable over Smith in view of Jones").
- **Allowable claims:** "objected to as being dependent upon a rejected base claim, but would be allowable if rewritten…".
- **Allowed claims:** "are allowed".

OARD's `objected` rows aren't used, because they also include objections for informalities.

| Decision class | Claims |
| --- | --- |
| 103 | 4,073 |
| 102 | 2,714 |
| 112 | 1,086 |
| 101 | 460 |
| No rejection | 324 (256 allowable dependent, 68 allowed) |

- **Multiple grounds:** 1,440 claims (21%) have grounds under more than one statute.
- **"In the alternative" statements:** "anticipated by or, in the alternative, obvious over X" gives both a 102 and a 103 ground.
- **Allowable over the art but still rejected under 112:** 18 claims ("would be allowable if rewritten to overcome the rejection under 35 U.S.C. 112").

**No-rejection examples are scarce.** Only 4.6% of claims in first actions are allowed or allowable. First actions reject most claims. Training needs class weighting or more negatives, for example from later actions in the same applications, where allowable subject matter is more common. That decision belongs to the SFT corpus build.

**OARD agreement.** OARD covers 6,663 of the claims, and its statutes match the parsed decision for 95.3% of them. The disagreements checked by hand were OARD errors. OARD adds the claims named in "as applied to claim 1 above" to its 103 claim lists. For example, "Claim 5 is rejected under 103 over Hamada as applied to claim 2 in view of Edmund" puts claim 2 in the 103 row as well.

**Grounds and references.** A 102/103 ground is valid when it resolves the references its wording implies: exactly one for 102, and one plus each "in view of" for 103. Of 9,094 grounds, 1,059 (11.6%) aren't valid. These fall into two groups:

- **Non-patent literature:** 293 claims. Journal articles named by title ("Higgins, 'Sickle cell vasoocclusion…'"). The ground is real but there's no document to fetch.
- **Unresolved:** 680 claims. Mostly a shorthand name ("…in view of Kobayashi") that the action never ties to a number. When an action has an unresolved ground, the builder tries to resolve the name against the first inventors of every document OARD lists for the application, including the examiner's PTO-892 form. Ambiguous names are dropped.

Fixing reference parsing along the way, with a regression test for each:

- patent numbers followed by a comma ("5,003,185, hereafter Burgio") were missed; this bug predates this work
- patent numbers with spaces after the commas ("6, 817, 711")
- a bare number after a name ("Spahrbier 3843410")
- design patents ("Des. 373,240")
- spaced EP numbers ("EP 1 192 959", "European Patent Application 1 455 128")
- lowercase cues ("US patent 6127444")
- statements cut off at initials ("in view of St. Germain", "F.C.")
- explicit numbers now take precedence over a name used for a different document elsewhere in the action
- priority applications ("claims foreign priority to EP 05001407.5") are no longer counted as references
- bracketed per-claim headings ("[Claim 5]") now start a claim's segment

## Claim text

| Source | Records |
| --- | --- |
| Pre-grant publication | 6,629 |
| Claims as filed (file-wrapper OCR), for unpublished applications | 376 |

Before this work, a claim counted as unreliable whenever *any* amendment preceded the action. That was too cautious: it covered 2,587 records (37%), most of them from preliminary amendments that never touched the claims. It's replaced by two checks:

1. **The amendment's status labels.** Any change to the claims has to come with a full claim listing that labels each claim "(original)", "(currently amended)", "(new)" and so on (37 CFR 1.121(c)). The file wrapper stores this listing as a later `CLM` document. "(original)" verifies the as-filed text (461 claims). Any other status marks it as not the claim examined (350 claims: 138 currently amended, 113 previously presented, 91 new, 8 withdrawn).
2. **The examiner's restatement.** Examiners usually restate the claim while mapping it to the art. When the examiner's text for that claim repeats at least half of the claim's word 4-grams (ignoring bracketed and parenthetical cites), the text is verified. Of 3,504 claims with a single-claim segment, the quartiles of overlap are 0.35, 0.73 and 0.90. Low overlap doesn't mark a claim unverified, because examiners often argue by reference ("arguments analogous to claim 1 apply").

When no listing was filed after the application, the claims as filed are the claims examined. Overall, claim text is reliable for 6,423 records (91.7%): 2,645 verified, plus 3,778 with no later claim listing.

The OCR'd claims carried page headers and line numbers in the text ("further 15 step"), and these are now stripped.

## Limitations and the examiner's reasoning

- **Limitations:** each claim is split into its own limitations, 1.65 per claim on average. An independent claim drops its preamble and splits at semicolons and before ", wherein". A dependent claim drops "The method of claim 1, further comprising". Every limitation is a verbatim substring of the claim.
- **The examiner's text for a claim** comes from:
  - "Regarding claim N" style segments (and "As to", "With respect to" and similar)
  - otherwise, sentences that name the claim ("[as in claims 3-4]", "addresses claim 19"; 1,753 claims)
  - otherwise, for 101/112, the rejection's whole argument, which usually covers the claim group (388 claims)
- **Allowable claims:** the explanation is the statement of reasons for allowable subject matter, when it names the claim (51 claims).
- **No examiner text:** 811 claims, most of them allowable claims, where examiners usually give no reasons.
- **Pin cites:** 12,246 in claims' reasoning.
  - 5,146 resolve to publication paragraphs.
  - 3,810 are matched to the limitation the examiner restated just before the cite, which needs 60% of the limitation's content words in that span.
  - On average, the pin cites cover 40% of an art-rejected claim's limitations.
  - Granted patents are cited by column and line, which the XML can't resolve.

## Quality tiers

| Tier | Rejected | Allowable dependent | Allowed | Total |
| --- | --- | --- | --- | --- |
| Grounded | 1,334 | 171 | 56 | 1,561 |
| Labeled | 3,915 | 66 | 5 | 3,986 |
| Weak | 1,432 | 19 | 7 | 1,458 |

- **Grounded:** reliable claim text, valid grounds, no conflicting dispositions, and cited art with full text. For a 102/103 rejection it also needs at least one resolved pin cite.
- **Labeled:** the same label quality, but missing the full text or the resolved evidence.
- **Weak:** anything else.

Validation and test sets take only grounded records.

## Evidence

**Documents.** The cited art across the 500 actions is 1,344 documents:

| Kind | Documents | With full text |
| --- | --- | --- |
| US publications | 647 | 504 |
| US patents | 632 | 186 |
| Foreign | 65 | 0 |

They split into 88,749 passages (abstract plus description paragraphs, median 83 per document, each capped at 300 tokens). Publications keep their printed paragraph numbers as locators ("[0092]"). Grant paragraphs get a position locator that can't be mistaken for an examiner cite.

Cited "patents" dated on or after the action are dropped as misread numbers. The one case found was an application serial number read as a patent number.

Document IDs are normalized to one form for both OARD and the office action text:
- US publications: 11 digits
- US patents: digits with no leading zeros, plus a D/RE/PP prefix where needed
- foreign documents: country code plus digits

Kind codes are dropped, and a test checks that the two sides agree on every format.

**Budget.**

- **Fixed limit:** 2,500 evidence tokens and 16 passages per claim.
- **Why this size:** with a 4,096-token context, the claim and its parents (237 tokens at the median, 549 at the 95th percentile) and 1,000 tokens of instructions and output still fit.
- **Fit:** 98.4% of records (6,893) fit within the limit.
- **Token counts:** estimated at 4 characters per token until the base model's tokenizer is fixed.

**Conditions.** Two evidence conditions use this budget. Ranking only ever uses the claim text.

- **`action_cited` (primary).** First, every passage the examiner pin-cited anywhere in the action, best match to the claim first, up to 70% of the budget. Then BM25 fills the rest: each document's best passage, the best two per limitation, then the best for the whole claim. The cited pool is the same for every claim in the action, rejected or allowable, so it shows what the examiner looked at but not the decision on this claim.
- **`retrieved`.** BM25 only, with no input from the office action. This is the realistic setting, and it's kept for the ablation.

**Recall.** This measures how often the examiner's own pin cites for the claim reach the input. It covers the 1,572 claims that have resolved pin cites.

| Evidence tokens | `action_cited`: claims with ≥1 cited passage | `action_cited`: share of cited passages | `retrieved`: claims with ≥1 | `retrieved`: share |
| --- | --- | --- | --- | --- |
| 500 | 59% | 23% | 23% | 7% |
| 1,000 | 74% | 36% | 34% | 12% |
| 2,000 | 88% | 54% | 45% | 18% |
| **2,500 (chosen)** | **91%** | **60%** | **49%** | **21%** |
| 3,000 | 95% | 65% | 54% | 25% |
| 4,000 | 97% | 72% | 60% | 30% |

Lexical retrieval alone finds an examiner-cited passage for under half the claims, at 1.7–2.2 times the rate of random selection on the 60-action development set. Querying each limitation separately helped a little, and adding neighboring paragraphs didn't help. If training used `retrieved` evidence, the model would often be taught to reproduce a rejection without seeing the passage that supports it. That is the unsupported assertion this project measures, so `action_cited` is the primary condition. Whether dense retrieval closes the gap is a later ablation.

**No passages at all:** 1,696 records (24%). The cited art is foreign, non-patent, or older patents without full text. None of these records is grounded.

## Next

- **SFT corpus:** build schema-valid `Example`s from grounded and labeled records. Map pin cites to passage IDs present in the input, and drop or flag mapping targets whose passage didn't fit. Decide how to rebalance the 4.6% no-rejection class.
- **Splits:** application families are still to be resolved. The records keep `application_id`.
