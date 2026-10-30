"""Download the Office Action Research Dataset zips from the USPTO Open Data Portal.

Requires USPTO_API_KEY. Alternatively, download office_actions.csv.zip, rejections.csv.zip
and citations.csv.zip from https://data.uspto.gov/bulkdata/datasets/ptoffact by hand and
place them in the zip_dir from the config.

Usage: uv run python scripts/download_oard.py [configs/oard_sample.yaml]
"""

import sys
from pathlib import Path

from patent_reasoning.config import load_config
from patent_reasoning.data.oard import download_tables


def main() -> None:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/oard_sample.yaml")
    for path in download_tables(Path(cfg.params["zip_dir"])):
        print(f"{path} ({path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
