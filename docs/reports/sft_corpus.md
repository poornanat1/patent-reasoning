# Initial SFT corpus

> **Superseded numbers.** Parser fixes from the dataset audit changed these counts; the current ones are in [dataset_v1.md](dataset_v1.md).

**What this covers.** Each claim record becomes a schema-valid `Example` (see [docs/schemas/](../schemas/)), but only when its target is grounded in what the model will see. Every other record is kept in an exclusion file, with the reason it was left out.

**Result.** 1,144 examples from 206 applications, out of 7,005 claim records (16%). None failed schema validation.

Raw numbers: [sft_corpus_summary.json](sft_corpus_summary.json). Pipeline: `scripts/build_sft_corpus.py` ([configs/sft_corpus.yaml](../../configs/sft_corpus.yaml)), with rules in `src/patent_reasoning/data/sft.py`. Output goes to `data/interim/sft_corpus/examples.jsonl` and `excluded.jsonl`.

## What qualifies

**Input.**
- the claim
- the claims it depends on, nearest first
- every document cited in the office action, with this claim's passages from the primary `action_cited` evidence condition

**Target.**

| Field | Source |
| --- | --- |
| `grounds` | The claim's rejection grounds, with duplicates removed |
| `claim_mapping` | Each of the claim's own limitations that the examiner tied to a pin cite, with the cited passages that are in the input. Passages outside the input are never cited. |
| `explanation` | The examiner's reasoning for the claim, or the reasons for allowable subject matter when they name the claim |
| `evidence_status` | `supported` when every limitation is mapped, `partially_supported` otherwise, `not_applicable` without a 102/103 ground |

**No-rejection claims.** An allowable dependent claim with a single own limitation maps that limitation as not disclosed, which is what the examiner concluded. Other no-rejection claims get no mapping (flag `mapping_unknown`), because the examiner doesn't say which limitation the art lacks.

**Exclusions.** A record is excluded, in this order, when:

| Reason | Records | Why |
| --- | --- | --- |
| `claim_text_unreliable` | 582 | The claim was amended before the action, so the text isn't what the examiner examined |
| `conflicting_disposition` | 32 | The action both rejects and allows the claim |
| `non_patent_literature` | 265 | A ground relies on a journal article, which has no document text |
| `references_unresolved` | 579 | A ground's reference couldn't be identified |
| `no_prior_art_text` | 1,204 | A 102/103 rejection or a no-rejection claim with no cited passages to judge against |
| `evidence_not_in_input` | 2,859 | A 102/103 rejection where none of the examiner's cited passages are in the input. 2,727 of these cite granted patents by column and line, which can't be resolved. In 132, the cited passages didn't fit the budget. |
| `explanation_off_claim` | 84 | The examiner text found for the claim restates none of its limitations. It's about another claim, e.g. "Shaknovich and Weber teach the limitations of claim 4, upon which claim 5 depends". |
| `target_too_long` | 256 | The target is over 700 tokens, which with the instructions would exceed the 1,000-token reserve |

The `evidence_not_in_input` rule is the one that matters most. Training on a rejection whose supporting passages the model never sees would teach it to assert disclosure without evidence. The project exists to measure exactly that failure.

## Composition

| Decision class | Examples |
| --- | --- |
| 103 | 546 |
| 102 | 290 |
| 112 | 143 |
| 101 | 68 |
| No rejection | 224 (168 allowable dependent, 56 allowed) |

**Negatives.** No-rejection claims are 19.6% of the corpus, against 4.6% of claim records. Rejections lose far more records to the evidence rules than allowable claims do, so the filter rebalances the classes as a side effect.

| Property | Count |
| --- | --- |
| Evidence status `supported` / `partially_supported` / `not_applicable` | 581 / 253 / 310 |
| With an explanation | 965 (919 examiner reasoning, 46 reasons for allowance) |
| Without an explanation | 179, mostly allowable claims; flagged `no_reference_explanation` |
| With a claim mapping | 777 examples, 1,024 mapped limitations |
| Partial mapping (`mapping_partial`) | 240 |
| OARD statutes disagree with the parsed decision | 41 |
| Target tokens (median / 95th percentile / max) | 224 / 595 / 699 |

## Flags that stay on included examples

These mark examples that are usable but less certain. Training and evaluation can filter on them:

- **`mapping_partial`:** some of the claim's limitations aren't mapped to a passage.
- **`mapping_unknown`:** a no-rejection claim where the examiner didn't say which limitation the art lacks.
- **`no_reference_explanation`:** the examiner gave no reasons, so the target explanation is null.
- **`reasoning_mention`:** the explanation comes from sentences that name the claim rather than a "Regarding claim N" segment.
- **`reasoning_shared`:** for 101/112, the rejection's whole argument, which usually covers the claim group.
- **`restatement_low_overlap`:** the examiner's text repeats less than half of the claim's word 4-grams.
- **`claims_as_filed_ocr`:** claim text comes from OCR'd as-filed claims, for unpublished applications.

## Fixes found while building it

- **Bracketed headings.** Some examiners head each claim with "[Claim 5]". Those headings weren't recognized, and the sentence splitter didn't break before "[". Claim 5's explanation came out as the sentence about claim 4. Bracketed headings now start a segment.
- **Misread serial numbers.** An application serial number (11/113,115) had been read as US patent 11,113,115, granted in 2021 and cited in a 2011 action. Cited documents dated on or after the action are now dropped.

## Gap to the target size

The spec aims for about 20,000 SFT claims, and the corpus is 1,144. Two levers would close the gap:

1. **More actions.** This corpus comes from 500 office actions. The 1% sample has about 11,500 eligible first actions, so at the same yield the full sample gives roughly 26,000 examples. Fetching them takes about 17 hours of API time at the unthrottled rate. The run is resumable, because the cache keeps what's already fetched.
2. **Column and line cites.** Granted patents cited by column and line make up 2,727 exclusions, the biggest single loss. The full-text XML has no column or line numbers, so resolving these cites needs the printed page layout.

The first lever is mechanical, and the plan's split day can start on this corpus while the fetch continues.
