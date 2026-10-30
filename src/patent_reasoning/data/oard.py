"""USPTO Office Action Research Dataset (OARD), 2017 release.

Three CSV tables joined on ``ifw_number`` (one row per office action document):

- ``office_actions``: one row per office action, with rejection flags and quality flags
- ``rejections``: one row per (office action, action_type/action_subtype) pair
- ``citations``: one row per application-citation pair, linked to an action when cited in it

Schema source: USPTO "List of variables" and "dataset schema" PDFs, v20171120.
Files are served by the Open Data Portal (product ``PTOFFACT``) and need an API key.
"""

from __future__ import annotations

import hashlib
import os
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import requests

PRODUCT_ID = "PTOFFACT"
API_BASE = "https://api.uspto.gov/api/v1/datasets/products"
TABLES = ("office_actions", "rejections", "citations")

COLUMNS: dict[str, tuple[str, ...]] = {
    "office_actions": (
        "app_id", "ifw_number", "document_cd", "mail_dt", "art_unit", "uspc_class",
        "uspc_subclass", "header_missing", "fp_missing", "rejection_fp_mismatch",
        "closing_missing", "rejection_101", "rejection_102", "rejection_103", "rejection_112",
        "rejection_dp", "objection", "allowed_claims", "cite102_gt1", "cite103_gt3",
        "cite103_eq1", "cite103_max", "signature_type",
    ),
    "rejections": (
        "app_id", "ifw_number", "action_type", "action_subtype", "claim_numbers",
        "alice_in", "bilski_in", "mayo_in", "myriad_in",
    ),
    "citations": (
        "app_id", "citation_pat_pgpub_id", "parsed", "form892", "form1449", "citation_in_oa",
        "ifw_number", "action_type", "action_subtype",
    ),
}  # fmt: skip

DOCUMENTED_ROWS = {
    "office_actions": 4_384_532,
    "rejections": 10_133_179,
    "citations": 58_862_278,
}

# Stata exports dates as e.g. "02jan2008"; accept ISO and US formats too.
MAIL_DT_SQL = (
    "COALESCE(TRY_STRPTIME(mail_dt, '%d%b%Y'), TRY_STRPTIME(mail_dt, '%Y-%m-%d'),"
    " TRY_STRPTIME(mail_dt, '%m/%d/%Y'))::DATE"
)


# Grammar assumed for rejections.claim_numbers, e.g. "1-5, 7, 9-12". The audit measures
# how often the real data matches it.
CLAIM_NUMBERS_RE = r"\s*\d+(\s*-\s*\d+)?(\s*[,;]\s*\d+(\s*-\s*\d+)?)*\s*"


def parse_claim_numbers(value: str | None) -> list[int] | None:
    """Expand "1-3, 5" to [1, 2, 3, 5]. Returns None if the value doesn't fit the grammar."""
    if value is None or not re.fullmatch(CLAIM_NUMBERS_RE, value):
        return None
    claims: set[int] = set()
    for token in re.split(r"[,;]", value):
        lo, _, hi = token.partition("-")
        start, end = int(lo), int(hi or lo)
        if end < start:
            return None
        claims.update(range(start, end + 1))
    return sorted(claims)


def zip_name(table: str) -> str:
    return f"{table}.csv.zip"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _iter_file_entries(obj: Any) -> Iterator[dict[str, Any]]:
    """Yield every dict in a product response that describes a file."""
    if isinstance(obj, dict):
        if "fileName" in obj:
            yield obj
        for v in obj.values():
            yield from _iter_file_entries(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_file_entries(v)


def find_download_uri(product: dict[str, Any], file_name: str) -> str:
    seen = []
    for entry in _iter_file_entries(product):
        seen.append(entry["fileName"])
        if entry["fileName"] == file_name:
            for key, value in entry.items():
                if "download" in key.lower() and isinstance(value, str):
                    return value
    raise KeyError(f"{file_name} not in product file list; saw {sorted(seen)}")


def download_tables(out_dir: Path, api_key: str | None = None, tables=TABLES) -> list[Path]:
    """Download the OARD CSV zips from the Open Data Portal. Skips files already present."""
    api_key = api_key or os.environ.get("USPTO_API_KEY")
    if not api_key:
        raise RuntimeError(
            "USPTO_API_KEY is not set. Get a key at https://data.uspto.gov (My ODP), or "
            f"download the zips manually from https://data.uspto.gov/bulkdata/datasets/ptoffact "
            f"into {out_dir}."
        )
    headers = {"X-API-KEY": api_key}
    out_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(
        f"{API_BASE}/{PRODUCT_ID}", params={"includeFiles": "true"}, headers=headers, timeout=60
    )
    resp.raise_for_status()
    product = resp.json()

    paths = []
    for table in tables:
        dest = out_dir / zip_name(table)
        if dest.exists() and zipfile.is_zipfile(dest):
            paths.append(dest)
            continue
        uri = find_download_uri(product, zip_name(table))
        tmp = dest.with_suffix(".part")
        with requests.get(uri, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        if not zipfile.is_zipfile(tmp):
            raise RuntimeError(f"{uri} did not return a zip file")
        tmp.rename(dest)
        paths.append(dest)
    return paths


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_tables(zip_dir: Path, csv_dir: Path, tables=TABLES) -> dict[str, Path]:
    """Extract each table's CSV from its zip. Skips tables already extracted."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for table in tables:
        dest = csv_dir / f"{table}.csv"
        if not dest.exists():
            with zipfile.ZipFile(zip_dir / zip_name(table)) as zf:
                members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
                if len(members) != 1:
                    raise RuntimeError(f"Expected one CSV in {table} zip, found {members}")
                with zf.open(members[0]) as src, open(dest, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
        out[table] = dest
    return out


# ---------------------------------------------------------------------------
# Loading and sampling
# ---------------------------------------------------------------------------


def connect(paths: dict[str, Path]) -> duckdb.DuckDBPyConnection:
    """Open DuckDB with one view per table, from CSV or Parquet.

    CSVs are read as all-VARCHAR so IDs keep leading zeros; the sample Parquet files
    inherit those VARCHAR columns.
    """
    con = duckdb.connect()
    for table, path in paths.items():
        quoted = "'" + str(path).replace("'", "''") + "'"
        reader = (
            f"read_parquet({quoted})"
            if Path(path).suffix == ".parquet"
            else f"read_csv({quoted}, header=true, all_varchar=true)"
        )
        con.execute(f"CREATE VIEW {table} AS SELECT * FROM {reader}")
    return con


def in_sample(app_id: str, rate: float, salt: str = "oard") -> bool:
    """Deterministic, order-independent membership test for application-level sampling."""
    digest = hashlib.sha256(f"{salt}:{app_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate


def build_sample(
    con: duckdb.DuckDBPyConnection,
    out_dir: Path,
    rate: float,
    year_min: int,
    year_max: int,
    salt: str = "oard",
) -> dict[str, int]:
    """Sample applications with an office action mailed in [year_min, year_max].

    Keeps every office action, rejection and citation row for the sampled applications
    and writes them to Parquet. Returns row counts per table.
    """
    app_ids = [
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT app_id FROM office_actions WHERE year({MAIL_DT_SQL}) BETWEEN ? AND ?",
            [year_min, year_max],
        ).fetchall()
    ]
    sampled = sorted(a for a in app_ids if in_sample(a, rate, salt))
    con.register("sample_apps", pa.table({"app_id": pa.array(sampled, pa.string())}))

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"cohort_apps": len(app_ids), "sampled_apps": len(sampled)}
    for table in TABLES:
        dest = out_dir / f"{table}.parquet"
        con.execute(
            f"COPY (SELECT t.* FROM {table} t SEMI JOIN sample_apps s USING (app_id)) "
            f"TO '{dest}' (FORMAT parquet)"
        )
        counts[table] = con.execute(f"SELECT count(*) FROM '{dest}'").fetchone()[0]
    return counts
