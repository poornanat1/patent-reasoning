"""Audit OARD schemas, identifiers, labels and data quality.

Runs on the full extracted CSVs by default, or on the sample with --sample. Writes a
Markdown report and the raw results as JSON.

Usage: uv run python scripts/audit_oard.py [--sample] [configs/oard_sample.yaml]
"""

import argparse
import json
from pathlib import Path

from patent_reasoning.config import load_config
from patent_reasoning.data.audit import render_markdown, run_audit
from patent_reasoning.data.oard import TABLES, connect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/oard_sample.yaml")
    parser.add_argument("--sample", action="store_true", help="audit the sample, not full data")
    parser.add_argument("--out", default="docs/reports")
    args = parser.parse_args()

    p = load_config(args.config).params
    if args.sample:
        paths = {t: Path(p["out_dir"]) / f"{t}.parquet" for t in TABLES}
        name, title = "oard_audit_sample", "OARD audit (sample)"
    else:
        paths = {t: Path(p["csv_dir"]) / f"{t}.csv" for t in TABLES}
        name, title = "oard_audit", "OARD audit (full 2017 release)"

    result = run_audit(connect(paths), full=not args.sample)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.md").write_text(render_markdown(result, title))
    (out / f"{name}.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"Wrote {out / name}.md")


if __name__ == "__main__":
    main()
