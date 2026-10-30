# OARD audit (full 2017 release)

## Schema and row counts

| table | rows | documented_rows | missing_columns | extra_columns |
| --- | --- | --- | --- | --- |
| office_actions | 4,384,532 | 4,384,532 | [] | [] |
| rejections | 10,133,179 | 10,133,179 | [] | [] |
| citations | 58,862,278 | 58,862,278 | [] | [] |

## Blank values (columns with any)

| table | column | blank_rate |
| --- | --- | --- |
| rejections | action_subtype | 0.3006 |
| citations | citation_pat_pgpub_id | 1.70e-08 |
| citations | parsed | 0.0065 |
| citations | ifw_number | 0.7907 |
| citations | action_type | 0.7907 |
| citations | action_subtype | 0.8149 |

## Identifiers and joins

| check | value |
| --- | --- |
| oa_app_id_not_8_digits | 0 |
| oa_duplicate_ifw_numbers | 0 |
| oa_distinct_apps | 2,188,039 |
| rej_orphan_ifw | 0 |
| rej_app_id_mismatch | 0 |
| cit_orphan_apps | 316,827 |
| cit_linked_rows | 12,320,882 |
| cit_linked_orphan_actions | 0 |

## Mail dates

| check | value |
| --- | --- |
| unparsed_rate | 0.0000 |
| min_date | 2001-01-01 |
| max_date | 2017-07-11 |
| examples | ['2015-10-15', '2015-10-15', '2015-10-15'] |

| year | office_actions | apps |
| --- | --- | --- |
| 2001 | 28 | 28 |
| 2008 | 6,837 | 6,583 |
| 2009 | 66,926 | 57,394 |
| 2010 | 205,596 | 167,979 |
| 2011 | 426,134 | 334,847 |
| 2012 | 573,016 | 442,412 |
| 2013 | 649,765 | 486,367 |
| 2014 | 677,561 | 509,661 |
| 2015 | 851,789 | 565,296 |
| 2016 | 748,472 | 543,159 |
| 2017 | 178,408 | 164,791 |

## Office action flag rates

| check | value |
| --- | --- |
| rejection_101 | 0.1093 |
| rejection_102 | 0.4193 |
| rejection_103 | 0.7917 |
| rejection_112 | 0.3484 |
| rejection_dp | 0.1071 |
| objection | 0.2686 |
| allowed_claims | 0.0715 |
| header_missing | 0.0707 |
| fp_missing | 0.1055 |
| rejection_fp_mismatch | 0.0480 |
| closing_missing | 0.0017 |

## Document codes

| document_cd | n |
| --- | --- |
| CTNF | 2,961,350 |
| CTFR | 1,423,182 |

## Rejection action types

| action_type | action_subtype | n |
| --- | --- | --- |
| 103 | a | 3,001,862 |
| 112 | b | 1,245,752 |
| 102 | b | 1,233,221 |
| objected |  | 1,177,245 |
| cancelled |  | 611,066 |
| 103 |  | 479,613 |
| nonstatutory double patenting |  | 456,595 |
| 112 | a | 438,046 |
| 102 | a | 336,892 |
| 102 | e | 298,987 |
| allowed |  | 278,348 |
| 101 | Non-Statutory-data | 124,216 |
| 101 | Non-Statutory-other | 99,414 |
| 112 | d | 71,121 |
| 101 | Non-Statutory-alice | 56,709 |
| 101 | Non-Statutory-nature | 48,094 |
| 101 | Non-Statutory-bilski | 31,143 |
| 101 | Statutory-double-patenting | 31,103 |
| 101 | Non-Statutory-mentally | 24,188 |
| 101 | Non-Statutory-software | 23,114 |
| 112 |  | 21,845 |
| 102 |  | 21,759 |
| 101 | Non-Statutory-mayo | 8,725 |
| 101 | Useful | 8,075 |
| 101 | Non-Statutory-myriad | 3,599 |

## Flag vs. rejection-row agreement

| flag | both | flag_only | rows_only |
| --- | --- | --- | --- |
| rejection_101 | 458,373 | 20,691 | 0 |
| rejection_102 | 1,819,164 | 19,370 | 0 |
| rejection_103 | 3,468,160 | 3,180 | 0 |
| rejection_112 | 1,483,062 | 44,587 | 0 |

## Claim numbers

| check | value |
| --- | --- |
| parse_rate | 1.0000 |
| unparsed_examples |  |

## Citations

| check | value |
| --- | --- |
| form892 | 0.3349 |
| form1449 | 0.6731 |
| citation_in_oa | 0.2093 |
| parsed_blank | 0.0065 |

| action_type | n |
| --- | --- |
| 103 | 10,200,151 |
| 102 | 2,120,731 |

| citation_pat_pgpub_id | parsed |
| --- | --- |
| 20040088473 | 20040088473 |
| 20030189860 | 20030189860 |
| 20050013154 | 20050013154 |
| 20050013154 | 20050013154 |
| 6883060 | 6883060 |
