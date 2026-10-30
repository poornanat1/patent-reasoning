# data/

Raw and processed data live here but are not committed. Scripts regenerate them from public USPTO sources.

Planned layout:

- `raw/`: unmodified downloads (Office Action Research Dataset, claims, PatEx, publications)
- `interim/`: joined and parsed records
- `processed/`: SFT / DPO / validation / test splits plus a manifest with source hashes
