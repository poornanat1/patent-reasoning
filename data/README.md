# data/

Raw and processed data live here but are not committed. Scripts regenerate them from public USPTO sources.

- `raw/oard/`: Office Action Research Dataset zips (`office_actions`, `rejections`, `citations`), with the extracted CSVs in `raw/oard/csv/` (about 6.7 GB unzipped)
- `raw/odp/`: cached Open Data Portal responses (application metadata, document lists, office action and claims XML, publication and grant full text)
- `interim/oard_sample/`: the application-level sample as Parquet, plus `manifest.json` with source SHA-256s, parameters and row counts
- `interim/joined_sample/`: joined records (`records.jsonl`, one per office action × 102/103 rejection), a per-action funnel, a summary, and the manual inspection sheet
- `processed/`: SFT / DPO / validation / test splits (later)

The OARD tables join on `ifw_number` (one per office action). Citations link to a specific rejection through `ifw_number` + `action_type` + `action_subtype`. Read every ID column as a string so `app_id` keeps its leading zeros.
