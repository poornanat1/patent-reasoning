# data/

Raw and processed data live here but are not committed. Scripts regenerate them from public USPTO sources.

- `raw/oard/`: Office Action Research Dataset zips (`office_actions`, `rejections`, `citations`), with the extracted CSVs in `raw/oard/csv/` (about 6.7 GB unzipped)
- `interim/oard_sample/`: the application-level sample as Parquet, plus `manifest.json` with source SHA-256s, parameters and row counts
- `processed/`: SFT / DPO / validation / test splits (later)

The OARD tables join on `ifw_number` (one per office action). Citations link to a specific rejection through `ifw_number` + `action_type` + `action_subtype`. Read every ID column as a string so `app_id` keeps its leading zeros.
