"""Unit 4.4 — CSV/XLSX imports recognise South African debit/credit headings.

Banks often label columns ``Debit (R)`` / ``Credit (R)`` (or plural / bracket
variants). Headings must normalise before pattern matching and amounts must
sign correctly (money in positive, money out negative). TEST data only; ingress
only — does not touch the frozen analysis engine.
"""
import io
import os
import tempfile
from datetime import datetime

import pytest

from bank_statements.excel_reader import BankStatementExcelReader


def _read(text, suffix=".csv", encoding="utf-8"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as handle:
        handle.write(text.encode(encoding) if isinstance(text, str) else text)
    try:
        reader = BankStatementExcelReader()
        df = reader.read_excel(path)
        return df, reader.get_errors()
    finally:
        os.remove(path)


def _amounts(df):
    return [round(float(a), 2) for a in df["Amount"]]


@pytest.mark.parametrize(
    "header_line,debit_cell,credit_cell,expected_amount",
    [
        ("Date,Description,Debit (R),Credit (R),Balance (R)", "5 000.00", "", -5000.0),
        ("Date,Description,Debits (R),Credits (R),Balance (R)", "250.50", "", -250.5),
        ("Date,Narrative,Withdrawal [R],Deposit [R],Balance", "", "1 200.00", 1200.0),
        ("Transaction Date,Details,Debit (ZAR),Credit (ZAR)", "", "99.99", 99.99),
    ],
)
def test_csv_sa_debit_credit_header_variants(
    header_line, debit_cell, credit_cell, expected_amount
):
    """Each variant must import one row with the correct sign."""
    csv_text = (
        f"{header_line}\n"
        f"03/04/2025,TEST PAYMENT,{debit_cell},{credit_cell},0\n"
    )
    df, errors = _read(csv_text)
    assert df is not None, errors
    assert len(df) == 1
    assert _amounts(df) == [expected_amount]


def test_csv_sa_headers_credit_then_debit_rows():
    """Classic Nedbank-style export: credit receipt then debit payment."""
    csv_text = (
        "Date,Description,Debit (R),Credit (R),Balance (R)\n"
        "2026-03-01,Sales,,15 000.00,15000.00\n"
        "2026-03-02,Rent,5 000.00,,10000.00\n"
    )
    df, errors = _read(csv_text)
    assert df is not None, errors
    assert _amounts(df) == [15000.0, -5000.0]
    assert df.iloc[0]["Description"] == "Sales"
    assert df.iloc[1]["Description"] == "Rent"


def test_xlsx_sa_debit_credit_headings():
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Standard Bank — TEST"])
    ws.append([])
    ws.append(["Date", "Description", "Debit (R)", "Credit (R)", "Balance (R)"])
    ws.append([datetime(2026, 3, 1), "Client deposit", None, 3200.5, 3200.5])
    ws.append([datetime(2026, 3, 2), "Supplier", 500.0, None, 2700.5])
    buf = io.BytesIO()
    wb.save(buf)
    df, errors = _read(buf.getvalue(), suffix=".xlsx")
    assert df is not None, errors
    assert _amounts(df) == [3200.5, -500.0]


def test_upload_route_accepts_csv_with_sa_debit_credit_headers(canary_app):
    """End-to-end through the upload endpoint users hit."""
    from models import db, User, Account, Transaction

    app = canary_app
    with app.app_context():
        user = User(
            username="sa-hdr",
            email="sa-hdr@example.com",
            subscription_status="active",
        )
        user.set_password("Sup3rSecret!")
        db.session.add(user)
        db.session.commit()
        bank = Account(
            name="Bank",
            link="ca.810.001",
            category="Assets",
            user_id=user.id,
            is_active=True,
        )
        db.session.add(bank)
        db.session.commit()
        user_id, account_id = user.id, bank.id

    client = app.test_client()
    resp = client.post(
        "/auth/login",
        data={"email": "sa-hdr@example.com", "password": "Sup3rSecret!"},
    )
    assert "/login" not in resp.headers.get("Location", "")

    csv_bytes = (
        "Date,Description,Debit (R),Credit (R),Balance (R)\n"
        "03/04/2025,POS PURCHASE,150.00,,850.00\n"
        "04/04/2025,SALARY,,15000.00,15850.00\n"
    ).encode()
    resp = client.post(
        "/bank-statements/upload",
        data={
            "account": str(account_id),
            "file": (io.BytesIO(csv_bytes), "sa-headers.csv"),
        },
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    body = resp.get_json()
    assert resp.status_code == 200, body
    assert body["success"] is True
    with app.app_context():
        txs = Transaction.query.filter_by(user_id=user_id).order_by(
            Transaction.date
        ).all()
        assert len(txs) == 2
        assert float(txs[0].amount) == pytest.approx(-150.0)
        assert float(txs[1].amount) == pytest.approx(15000.0)
