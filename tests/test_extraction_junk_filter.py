"""Unit 1.7 — extraction junk-row filter (statement notices, not transactions)."""
from decimal import Decimal

import pandas as pd
import pytest

from ocr.bank_profiles import parse_transaction_lines
from services.extraction_junk_filter import (
    JunkVerdict,
    classify_extraction_row,
    filter_normalized_rows,
    filter_review_rows,
)


@pytest.mark.parametrize(
    "description",
    [
        "Schd Trxn No Av Bal",
        "SCHEDULED TRXN",
        "Available balance",
        "No Av Bal",
        "Balance summary for account",
    ],
)
def test_zero_amount_notice_lines_are_dropped(description):
    assert classify_extraction_row(description, "0.00") == JunkVerdict.DROP
    assert classify_extraction_row(description, 0) == JunkVerdict.DROP


def test_non_zero_amounts_are_never_dropped_even_with_notice_wording():
    verdict = classify_extraction_row("Schd Trxn No Av Bal", "12.50")
    assert verdict == JunkVerdict.KEEP


def test_ambiguous_zero_amount_is_flagged_not_dropped():
    assert classify_extraction_row("Bank fee reversal pending", "0.00") == JunkVerdict.FLAG


def test_filter_review_rows_drops_junk_and_flags_other_zeros():
    rows = [
        {"date": "2026-03-01", "description": "Schd Trxn No Av Bal", "amount": 0.0, "confidence": 0.9},
        {"date": "2026-03-02", "description": "CARD PURCHASE", "amount": -50.0, "confidence": 0.9},
        {"date": "2026-03-03", "description": "Fee waiver", "amount": 0.0, "confidence": 0.9},
    ]
    kept, dropped = filter_review_rows(rows)
    assert dropped == 1
    assert len(kept) == 2
    assert kept[0]["description"] == "CARD PURCHASE"
    assert kept[0]["statement_notice"] is False
    assert kept[1]["description"] == "Fee waiver"
    assert kept[1]["statement_notice"] is True


def test_pdf_text_parser_skips_scheduled_transaction_notice():
    text = """
Opening Balance 1 000.00
15/03/2026 Schd Trxn No Av Bal 0.00 1 000.00
20/03/2026 SALARY DEPOSIT 2 500.00 3 500.00
Closing Balance 3 500.00
"""
    lines = parse_transaction_lines(text)
    assert len(lines) == 1
    assert lines[0].description == "SALARY DEPOSIT"
    assert lines[0].amount == Decimal("2500.00")


def test_csv_normalized_rows_drop_notice_lines():
    rows = [
        {"Date": pd.Timestamp("2026-03-01"), "Description": "Schd Trxn No Av Bal", "Amount": 0.0},
        {"Date": pd.Timestamp("2026-03-02"), "Description": "EFT Payment", "Amount": -100.0},
        {"Date": pd.Timestamp("2026-03-03"), "Description": "Rounding adjustment", "Amount": 0.0},
    ]
    kept, dropped, flagged = filter_normalized_rows(rows)
    assert dropped == 1
    assert flagged == 1
    assert len(kept) == 2
    assert kept[0]["Description"] == "EFT Payment"
    assert kept[1]["Description"] == "Rounding adjustment"


def test_excel_reader_filters_junk_rows(tmp_path):
    from bank_statements.excel_reader import BankStatementExcelReader

    path = tmp_path / "notice.csv"
    path.write_text(
        "Date,Description,Amount\n"
        "2026-03-01,Schd Trxn No Av Bal,0.00\n"
        "2026-03-02,Supplier payment,-150.00\n",
        encoding="utf-8",
    )
    reader = BankStatementExcelReader()
    df = reader.read_excel(str(path))
    assert df is not None
    assert len(df) == 1
    assert df.iloc[0]["Description"] == "Supplier payment"
