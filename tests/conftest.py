"""Small OARD fixture that follows the documented 2017 schema."""

import csv
import io
import zipfile
from pathlib import Path

import pytest

from patent_reasoning.data import oard

OA_FLAGS = {c: "0" for c in oard.COLUMNS["office_actions"][7:]}


def _oa(app_id, ifw, mail_dt, **flags):
    row = dict.fromkeys(oard.COLUMNS["office_actions"], "")
    row.update(OA_FLAGS, app_id=app_id, ifw_number=ifw, document_cd="CTNF", mail_dt=mail_dt)
    row.update(flags)
    return row


def _rej(app_id, ifw, action_type, claims):
    row = dict.fromkeys(oard.COLUMNS["rejections"], "0")
    row.update(app_id=app_id, ifw_number=ifw, action_type=action_type, action_subtype="",
               claim_numbers=claims)  # fmt: skip
    return row


def _cit(app_id, cited, ifw="", action_type="", in_oa="0"):
    return dict(app_id=app_id, citation_pat_pgpub_id=cited, parsed=cited, form892="1",
                form1449="0", citation_in_oa=in_oa, ifw_number=ifw, action_type=action_type,
                action_subtype="")  # fmt: skip


FIXTURE = {
    "office_actions": [
        _oa("01234567", "IFW1", "02jan2009", rejection_103="1"),
        _oa("01234567", "IFW2", "15mar2010", rejection_102="1"),
        _oa("12345678", "IFW3", "01jun2012", rejection_101="1"),
        _oa("23456789", "IFW4", "05may2016", rejection_103="1"),  # outside cohort
    ],
    "rejections": [
        _rej("01234567", "IFW1", "103", "1-3, 5"),
        _rej("01234567", "IFW2", "102", "1,2"),
        _rej("12345678", "IFW3", "101", "claims 1 and 4"),
        _rej("23456789", "IFW4", "103", "7"),
        _rej("99999999", "IFW9", "103", "1"),  # orphan action
    ],
    "citations": [
        _cit("01234567", "US7654321", "IFW1", "103", "1"),
        _cit("01234567", "US20080012345"),
        _cit("12345678", "US6543210"),
        _cit("23456789", "US5432109", "IFW4", "103", "1"),
    ],
}


def _write_zips(zip_dir: Path) -> None:
    zip_dir.mkdir(parents=True)
    for table, rows in FIXTURE.items():
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=oard.COLUMNS[table])
        writer.writeheader()
        writer.writerows(rows)
        with zipfile.ZipFile(zip_dir / oard.zip_name(table), "w") as zf:
            zf.writestr(f"{table}.csv", buf.getvalue())


@pytest.fixture
def csvs(tmp_path):
    _write_zips(tmp_path / "zips")
    return oard.extract_tables(tmp_path / "zips", tmp_path / "csv")
