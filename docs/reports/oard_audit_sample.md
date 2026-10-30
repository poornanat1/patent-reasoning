# OARD audit (sample)

## Schema and row counts

| table | rows | documented_rows | missing_columns | extra_columns |
| --- | --- | --- | --- | --- |
| office_actions | 30,693 |  | [] | [] |
| rejections | 69,823 |  | [] | [] |
| citations | 381,364 |  | [] | [] |

## Blank values (columns with any)

| table | column | blank_rate |
| --- | --- | --- |
| rejections | action_subtype | 0.2592 |
| citations | parsed | 0.0074 |
| citations | ifw_number | 0.7709 |
| citations | action_type | 0.7709 |
| citations | action_subtype | 0.7744 |

## Identifiers and joins

| check | value |
| --- | --- |
| oa_app_id_not_8_digits | 0 |
| oa_duplicate_ifw_numbers | 0 |
| oa_distinct_apps | 14,247 |
| rej_orphan_ifw | 0 |
| rej_app_id_mismatch | 0 |
| cit_orphan_apps | 0 |
| cit_linked_rows | 87,356 |
| cit_linked_orphan_actions | 0 |

## Mail dates

| check | value |
| --- | --- |
| unparsed_rate | 0.0000 |
| min_date | 2008-04-02 |
| max_date | 2017-07-11 |
| examples | ['2009-02-10', '2009-02-11', '2009-02-12'] |

| year | office_actions | apps |
| --- | --- | --- |
| 2008 | 67 | 65 |
| 2009 | 658 | 563 |
| 2010 | 2,033 | 1,687 |
| 2011 | 4,283 | 3,340 |
| 2012 | 5,678 | 4,380 |
| 2013 | 6,709 | 5,026 |
| 2014 | 6,972 | 5,249 |
| 2015 | 3,082 | 1,996 |
| 2016 | 1,056 | 731 |
| 2017 | 155 | 143 |

## Office action flag rates

| check | value |
| --- | --- |
| rejection_101 | 0.1002 |
| rejection_102 | 0.4152 |
| rejection_103 | 0.7943 |
| rejection_112 | 0.3366 |
| rejection_dp | 0.0971 |
| objection | 0.2601 |
| allowed_claims | 0.0729 |
| header_missing | 0.0711 |
| fp_missing | 0.1097 |
| rejection_fp_mismatch | 0.0440 |
| closing_missing | 0.0020 |

## Document codes

| document_cd | n |
| --- | --- |
| CTNF | 20,273 |
| CTFR | 10,420 |

## Rejection action types

| action_type | action_subtype | n |
| --- | --- | --- |
| 103 | a | 23,984 |
| 102 | b | 9,844 |
| 112 | b | 8,300 |
| objected |  | 7,979 |
| cancelled |  | 4,523 |
| 112 | a | 3,005 |
| nonstatutory double patenting |  | 2,890 |
| 102 | e | 2,545 |
| allowed |  | 1,976 |
| 101 | Non-Statutory-data | 935 |
| 102 | a | 722 |
| 101 | Non-Statutory-other | 537 |
| 103 |  | 448 |
| 112 | d | 369 |
| 101 | Non-Statutory-bilski | 290 |
| 101 | Non-Statutory-nature | 274 |
| 101 | Non-Statutory-alice | 274 |
| 112 |  | 220 |
| 101 | Statutory-double-patenting | 215 |
| 101 | Non-Statutory-mentally | 147 |
| 101 | Non-Statutory-software | 140 |
| 102 |  | 64 |
| 101 | Useful | 62 |
| 101 | Non-Statutory-mayo | 48 |
| 101 | Non-Statutory-myriad | 18 |

## Flag vs. rejection-row agreement

| flag | both | flag_only | rows_only |
| --- | --- | --- | --- |
| rejection_101 | 2,939 | 135 | 0 |
| rejection_102 | 12,630 | 115 | 0 |
| rejection_103 | 24,363 | 17 | 0 |
| rejection_112 | 10,017 | 314 | 0 |

## Claim numbers

| check | value |
| --- | --- |
| parse_rate | 1.0000 |
| unparsed_examples |  |

## Citations

| check | value |
| --- | --- |
| form892 | 0.3743 |
| form1449 | 0.6379 |
| citation_in_oa | 0.2291 |
| parsed_blank | 0.0074 |

| action_type | n |
| --- | --- |
| 103 | 72,588 |
| 102 | 14,768 |

| citation_pat_pgpub_id | parsed |
| --- | --- |
| 20080250914 | 20080250914 |
| 20070095297 | 20070095297 |
| 20090212123 | 20090212123 |
| 4863737 | 4863737 |
| 6267296 | 6267296 |
