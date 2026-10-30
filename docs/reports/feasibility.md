# Feasibility report: joining labels, office action text, claims and prior art

**Question.** Can USPTO examination records be joined into traceable examples? Each example needs the rejection label, the examiner's reasoning, the rejected claims' text and the cited prior art's text. And can enough of them support a claim-level, evidence-grounded benchmark?

**Answer.** Yes, with a narrower evidence scope than planned.

- **Traceable records:** 163 from 200 office actions (target: 100+).
- **Strict evidence-grounded records:** 48 (target: 50+). These also have as-filed claims and at least one examiner pin cite resolved to the cited publication's paragraph text.
- **Decision: go.** Keep full reasoning generation. Build evidence-grounded targets from rejections that cite US publications from 2005 onward. Treat older patents as document-level evidence.

Raw numbers: [joined_sample_summary.json](joined_sample_summary.json). Pipeline: `scripts/build_joined_sample.py` (config [configs/joined_sample.yaml](../../configs/joined_sample.yaml)).

## Sources actually used

The plan assumed the Office Action Text Retrieval API and the Patent Claims Research Dataset. The Open Data Portal (ODP) file wrapper turned out to be a better source for both. Each `documentIdentifier` in the file wrapper equals the dataset's `ifw_number`, so joins are exact.

| Need | Source | Notes |
| --- | --- | --- |
| Labels, rejected claims, cited documents | Office Action Research Dataset (2017) | Audited in [oard_audit_findings.md](oard_audit_findings.md) |
| Office action text | ODP file wrapper → document `xmlarchive` | Structured XML: numbered form paragraphs plus free examiner text |
| Claims as filed | ODP `APPXML-SPLT` publication XML; as-filed `CLM` XML (OCR) when unpublished | Claims with numbers and dependencies |
| Amendment history | ODP file wrapper document list | Later `CLM` documents and `A.PE` preliminary amendments |
| Prior-art text | ODP application search → `APPXML-SPLT` / `PTGRXML-SPLT` | Paragraph-numbered full text |

PatentSearch (PatentsView) no longer resolves. Amended claims in the file wrapper (`CLM` XML) are OCR of the amendment markup, with deletions and insertions run together, so they aren't usable as text.

## Sample and funnel

200 non-final rejections (CTNF) mailed 2008–2014, drawn deterministically from the 1% application sample. Each has at least one 102 or 103 rejection row and clean OARD parsing flags. The candidate pool behind the sample is 13,748 actions, about 1.37M in the full dataset.

| Stage | Count |
| --- | --- |
| Office actions | 200 |
| Office action found in file wrapper, with XML text | 200 (100%) |
| Application has published claims | 191 (96%) |
| 102/103 rejection records (one per action × statute) | 280 (119 × 102, 161 × 103) |
| Rejection statement found in the text | 272 (97%) |
| All rejected claims found in the claim text | 257 (92%) |
| At least one cited US reference with full text | 181 (65%) |
| **Traceable** (statement + claims + prior-art text) | **163 (58%)** |
| Claims not amended before the action | 126 (45%) |
| At least one paragraph pin cite resolved to text | 101 (36%) |
| **Evidence-grounded** (traceable + as-filed claims + resolved paragraph evidence) | **48 (17%)**: 30 × 103, 18 × 102, across 39 applications |

Restricting to each application's first office action raises the strict yield to 47 of 217 records (22%). Later actions almost always follow an amendment.

## What limits evidence grounding

### 1. Full text exists only for prior art published from about 2005

| Cited US document | Unique docs | With full text |
| --- | --- | --- |
| Publications 2005+ | 196 | 196 (100%) |
| Publications 2001–2004 | 54 | 0 |
| Patents 7,000,000+ | 74 | 73 |
| Patents 6.0–6.99M | 95 | 12 |
| Patents below 6,000,000 | 134 | 0 (not in ODP) |

Overall, 281 of 553 unique US references (51%) have full text. Another 47 references are foreign (EP, WO, JP, KR, DE, ...), and examiners also cite non-patent literature that the dataset doesn't track. Bulk grant archives (APS/SGML/XML, 1976+) could fill the patent gap. But see point 2.

### 2. Granted patents are cited by column and line, which the XML can't resolve

Of 3,035 pin cites:

- **Column/line:** 1,297. 1,173 of these point at granted patents.
- **Paragraph:** 1,738.

Full-text XML has no column or line layout, so column/line cites can only be resolved from the printed page images, which would need OCR with column detection. For now, granted-patent references are document-level evidence. Paragraph cites resolve cleanly against publications:

| Paragraph pin cites | Count |
| --- | --- |
| Resolved to paragraph text | 1,222 (70%) |
| Publication without full text (2001–2004) | 282 |
| Not attributable to a document | 65 |
| Paragraph number not in the publication | 61 |
| "Paragraph" cite to a granted patent (ambiguous; not resolved) | 59 |
| Foreign document | 33 |
| Cites the applicant's own specification | 16 |

### 3. The amendment check is conservative

An action counts as "amended" if any claim listing was filed after the filing date and before the action. That includes elections after restriction requirements, which often only withdraw claims, and national-stage preliminary amendments. 89 of 217 first actions are flagged this way. The examiner's reasoning restates claim language closely: a median of about 90% of claim words appear in it, for amended and unamended records alike. That makes it possible to verify that published claims still match, instead of discarding every amended record.

## Manual review (50 traceable records)

I read 50 traceable records drawn deterministically (`scripts/inspection_sheet.py`). For each record I compared the rejection statement and claims against the OARD labels. For each resolved pin cite I read the cited paragraph against the examiner's sentence. The first pass found parser bugs, which were fixed before the counts above (see *Parser fixes*). Second-pass verdicts:

| Evidence check (26 records with resolved cites, 47 cites) | Cites | Records |
| --- | --- | --- |
| Paragraph clearly supports the examiner's statement | 28 | 17 |
| Related but only partly supports it | 13 | 6 |
| Doesn't support it or is unclear | 4 | 2 |
| Misattributed (cited the applicant's spec; now tagged `SELF`) | 2 | 1 |

**What I found in the other 24 records:**

- **Old or foreign references:** cited only by column/line to granted patents (8), or relying on pre-2005 or foreign references with no text (5).
- **No pin cites:** the examiner used only figure and reference-numeral mappings (6).
- **No real reasoning:** one rejection is entirely "See the reasons presented in the Written Opinion of the ISA", and one relies on non-patent literature.
- **Unnamed reference:** one combination names a reference ("Challenger") whose number appears only elsewhere in the action.

**General observations:**

- **Statutes:** the OARD statute matched the parsed statement in every reviewed record.
- **Claim lists:** these did not always match. Measured over all 272 records with a statement:

  | OARD claims vs. claims named in the statements | Records |
  | --- | --- |
  | Same | 200 |
  | OARD is missing claims the statement names | 49 |
  | OARD has extra claims | 19 |
  | Differ both ways | 4 |

  The misses come from OARD's own text parsing. For example, "Claims 1-6 and 8-15 are rejected under 102(b)" is stored as `1,2,3,4,5,6,23,24,25`. **Claim-level labels should come from the parsed statements, with OARD as a cross-check.**
- **Claim text:** where the examiner restates the claim, the as-filed claim text matched in the as-filed records I compared. Two records used the OCR fallback; both were clean enough to read.
- **Examiner style:** examiners write in very different styles. Some quote the reference verbatim ("Matsugu, ¶71"); others cite broad ranges ("paragraphs 110-118") or background paragraphs. "Partly supports" is common and is a property of the data, not of the join.

## Parser fixes made during review

- **Reference numbers:** now accepted in more forms ("US Patent #7679495", "USPN", "U.S. PG Pub."). Foreign numbers with commas or hyphens (EP 1,621,137; KR10-2004-0042965) no longer leak spurious US numbers.
- **Aliases:** now handle name suffixes (Jr., III), "… A1 to Becker", and document-word prefixes ("Rouse US Patent Application No. US …"). "USPG", "PG" and "EP" are no longer taken as names.
- **Statements:** accept "&", "Claim(s)" and "is/are".
- **Pin cites:** accept "par.", "(0045)" and "7:11-35". Cites after "Spec." are attributed to the application itself.
- **Grant paragraphs:** paragraph cites resolve only against publications.
- **Text cleanup:** image placeholders are stripped from the office action text.

**Known gaps:** paragraph lists ("par. 11, 13, 50" captures only the first) and aliases introduced as "by Jack J. LeTourneau (hereafter Letourneau)".

## Scale-up estimate

Per office action, the pipeline yields 0.82 traceable records and 0.24 strict evidence-grounded records, at about 8 API requests. Prior-art lookups are cached across actions.

- **2,500 evidence-grounded records** (validation, in-distribution test and OOD test): about 10,000 office actions and about 80k requests. That's roughly 8–10 hours at the current throttle of 0.35 s per request.
- **20,000 traceable SFT records:** about 25,000 office actions and up to about 200k requests.

The pool of about 1.37M candidate actions is not a constraint. Selecting first actions and rejections that cite 2005+ publications should raise the per-action yield.

## Decision and scope for dataset construction

1. **Go.** Build on the file-wrapper join. Take the statute from OARD rejection rows and the rejected claim numbers from the parsed statements, since OARD claim lists are incomplete in about 18% of records. Use the examiner's reasoning from the office action XML, and claims from the publication.
2. **Evidence-grounded targets** (claim mapping with paragraph citations): 102/103 rejections whose references include 2005+ US publications. Validation and test sets come only from strictly grounded records.
3. **Granted-patent and foreign references:** document-level evidence only. The model may name them but can't be scored on passage support.
4. **101/112 rejections:** classification plus examiner reasoning, with no evidence target.
5. **Before building splits:** prefer first actions, and replace the conservative amendment flag with a check that the examiner's claim restatement matches the published claim text.
