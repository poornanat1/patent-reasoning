"""Schema and data-quality audit for the Office Action Research Dataset.

Runs against the DuckDB views created by :func:`patent_reasoning.data.oard.connect`,
either on the full CSVs or on the sampled Parquet files.
"""

from __future__ import annotations

from typing import Any

import duckdb

from patent_reasoning.data.oard import CLAIM_NUMBERS_RE, COLUMNS, DOCUMENTED_ROWS, MAIL_DT_SQL

OA_FLAGS = (
    "rejection_101", "rejection_102", "rejection_103", "rejection_112", "rejection_dp",
    "objection", "allowed_claims",
)  # fmt: skip
OA_QUALITY_FLAGS = ("header_missing", "fp_missing", "rejection_fp_mismatch", "closing_missing")
# office_actions flag -> rejections.action_type value it should agree with
FLAG_TO_ACTION_TYPE = {
    "rejection_101": "101",
    "rejection_102": "102",
    "rejection_103": "103",
    "rejection_112": "112",
}


def _rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    cur = con.execute(sql)
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> Any:
    return con.execute(sql).fetchone()[0]


def _blank(col: str) -> str:
    return f"({col} IS NULL OR trim({col}) = '')"


def audit_schema(con, full: bool) -> list[dict[str, Any]]:
    out = []
    for table, expected in COLUMNS.items():
        actual = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
        rows = _scalar(con, f"SELECT count(*) FROM {table}")
        out.append(
            {
                "table": table,
                "rows": rows,
                "documented_rows": DOCUMENTED_ROWS[table] if full else None,
                "missing_columns": sorted(set(expected) - set(actual)),
                "extra_columns": sorted(set(actual) - set(expected)),
            }
        )
    return out


def audit_blanks(con) -> list[dict[str, Any]]:
    out = []
    for table, cols in COLUMNS.items():
        exprs = ", ".join(f"avg({_blank(c)}::INT) AS {c}" for c in cols)
        rates = _rows(con, f"SELECT {exprs} FROM {table}")[0]
        out += [{"table": table, "column": c, "blank_rate": rates[c]} for c in cols]
    return out


def audit_identifiers(con) -> dict[str, Any]:
    return {
        "oa_app_id_not_8_digits": _scalar(
            con, "SELECT count(*) FROM office_actions WHERE NOT regexp_full_match(app_id, '\\d{8}')"
        ),
        "oa_duplicate_ifw_numbers": _scalar(
            con, "SELECT count(*) - count(DISTINCT ifw_number) FROM office_actions"
        ),
        "oa_distinct_apps": _scalar(con, "SELECT count(DISTINCT app_id) FROM office_actions"),
        "rej_orphan_ifw": _scalar(
            con,
            "SELECT count(*) FROM rejections r ANTI JOIN office_actions o USING (ifw_number)",
        ),
        "rej_app_id_mismatch": _scalar(
            con,
            "SELECT count(*) FROM rejections r JOIN office_actions o USING (ifw_number) "
            "WHERE r.app_id <> o.app_id",
        ),
        "cit_orphan_apps": _scalar(
            con, "SELECT count(*) FROM citations c ANTI JOIN office_actions o USING (app_id)"
        ),
        "cit_linked_rows": _scalar(
            con, f"SELECT count(*) FROM citations WHERE NOT {_blank('ifw_number')}"
        ),
        "cit_linked_orphan_actions": _scalar(
            con,
            # Blank subtypes are NULL on both sides, so match them with IS NOT DISTINCT FROM.
            f"SELECT count(*) FROM (SELECT * FROM citations WHERE NOT {_blank('ifw_number')}) c "
            "ANTI JOIN rejections r ON c.ifw_number = r.ifw_number "
            "AND c.action_type = r.action_type "
            "AND c.action_subtype IS NOT DISTINCT FROM r.action_subtype",
        ),
    }


def audit_dates(con) -> dict[str, Any]:
    stats = _rows(
        con,
        f"SELECT avg(({MAIL_DT_SQL}) IS NULL ::INT) AS unparsed_rate, "
        f"min({MAIL_DT_SQL}) AS min_date, max({MAIL_DT_SQL}) AS max_date FROM office_actions",
    )[0]
    stats["examples"] = [
        r[0] for r in con.execute("SELECT mail_dt FROM office_actions LIMIT 3").fetchall()
    ]
    stats["by_year"] = _rows(
        con,
        f"SELECT year({MAIL_DT_SQL})::VARCHAR AS year, count(*) AS office_actions, "
        "count(DISTINCT app_id) AS apps FROM office_actions GROUP BY 1 ORDER BY 1",
    )
    return stats


def audit_labels(con) -> dict[str, Any]:
    flag_rates = ", ".join(f"avg(TRY_CAST({f} AS INT)) AS {f}" for f in OA_FLAGS + OA_QUALITY_FLAGS)
    agreement = []
    for flag, action_type in FLAG_TO_ACTION_TYPE.items():
        agreement.append(
            _rows(
                con,
                f"""
                WITH r AS (SELECT DISTINCT ifw_number FROM rejections
                           WHERE trim(action_type) = '{action_type}')
                SELECT '{flag}' AS flag,
                       sum((o.{flag} = '1' AND r.ifw_number IS NOT NULL)::INT) AS both,
                       sum((o.{flag} = '1' AND r.ifw_number IS NULL)::INT) AS flag_only,
                       sum((o.{flag} <> '1' AND r.ifw_number IS NOT NULL)::INT) AS rows_only
                FROM office_actions o LEFT JOIN r USING (ifw_number)
                """,
            )[0]
        )
    return {
        "oa_flag_rates": _rows(con, f"SELECT {flag_rates} FROM office_actions")[0],
        "document_cd": _rows(
            con,
            "SELECT document_cd, count(*) AS n FROM office_actions GROUP BY 1 ORDER BY n DESC "
            "LIMIT 15",
        ),
        "action_types": _rows(
            con,
            "SELECT action_type, action_subtype, count(*) AS n FROM rejections "
            "GROUP BY 1, 2 ORDER BY n DESC LIMIT 25",
        ),
        "flag_vs_rejection_rows": agreement,
        "claim_numbers_parse_rate": _scalar(
            con,
            f"SELECT avg(regexp_full_match(claim_numbers, '{CLAIM_NUMBERS_RE}')::INT) "
            f"FROM rejections WHERE NOT {_blank('claim_numbers')}",
        ),
        "claim_numbers_unparsed_examples": [
            r[0]
            for r in con.execute(
                f"SELECT DISTINCT claim_numbers FROM rejections "
                f"WHERE NOT {_blank('claim_numbers')} "
                f"AND NOT regexp_full_match(claim_numbers, '{CLAIM_NUMBERS_RE}') LIMIT 10"
            ).fetchall()
        ],
    }


def audit_citations(con) -> dict[str, Any]:
    return {
        "rates": _rows(
            con,
            "SELECT avg(TRY_CAST(form892 AS INT)) AS form892, "
            "avg(TRY_CAST(form1449 AS INT)) AS form1449, "
            "avg(TRY_CAST(citation_in_oa AS INT)) AS citation_in_oa, "
            f"avg({_blank('parsed')}::INT) AS parsed_blank FROM citations",
        )[0],
        "by_action_type": _rows(
            con,
            "SELECT action_type, count(*) AS n FROM citations "
            "WHERE TRY_CAST(citation_in_oa AS INT) = 1 GROUP BY 1 ORDER BY n DESC LIMIT 10",
        ),
        "examples": _rows(
            con,
            "SELECT citation_pat_pgpub_id, parsed FROM citations "
            "WHERE TRY_CAST(citation_in_oa AS INT) = 1 LIMIT 5",
        ),
    }


def run_audit(con: duckdb.DuckDBPyConnection, full: bool) -> dict[str, Any]:
    return {
        "schema": audit_schema(con, full),
        "blanks": audit_blanks(con),
        "identifiers": audit_identifiers(con),
        "dates": audit_dates(con),
        "labels": audit_labels(con),
        "citations": audit_citations(con),
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.2e}" if 0 < abs(v) < 1e-4 else f"{v:.4f}"
    if isinstance(v, int):
        return f"{v:,}"
    return "" if v is None else str(v)


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_none_\n"
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    lines += ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _kv(d: dict[str, Any]) -> str:
    return _table([{"check": k, "value": v} for k, v in d.items()])


def render_markdown(result: dict[str, Any], title: str) -> str:
    dates = {k: v for k, v in result["dates"].items() if k != "by_year"}
    labels = result["labels"]
    blanks = [r for r in result["blanks"] if r["blank_rate"]]
    parts = [
        f"# {title}\n",
        "## Schema and row counts\n",
        _table(result["schema"]),
        "## Blank values (columns with any)\n",
        _table(blanks),
        "## Identifiers and joins\n",
        _kv(result["identifiers"]),
        "## Mail dates\n",
        _kv(dates),
        _table(result["dates"]["by_year"]),
        "## Office action flag rates\n",
        _kv(labels["oa_flag_rates"]),
        "## Document codes\n",
        _table(labels["document_cd"]),
        "## Rejection action types\n",
        _table(labels["action_types"]),
        "## Flag vs. rejection-row agreement\n",
        _table(labels["flag_vs_rejection_rows"]),
        "## Claim numbers\n",
        _kv(
            {
                "parse_rate": labels["claim_numbers_parse_rate"],
                "unparsed_examples": "; ".join(labels["claim_numbers_unparsed_examples"]),
            }
        ),
        "## Citations\n",
        _kv(result["citations"]["rates"]),
        _table(result["citations"]["by_action_type"]),
        _table(result["citations"]["examples"]),
    ]
    return "\n".join(parts)
