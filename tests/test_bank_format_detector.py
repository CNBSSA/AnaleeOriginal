"""Tests for SA bank statement format detection."""
import pandas as pd
import pytest

from bank_statements.format_detector import normalize_bank_statement_dataframe


def test_standard_amount_columns():
    raw = pd.DataFrame([
        ['Date', 'Description', 'Amount'],
        ['2024-01-15', 'Rent payment', '-1500.00'],
        ['2024-01-20', 'Client deposit', '3200.50'],
    ])
    df = normalize_bank_statement_dataframe(raw)
    assert len(df) == 2
    assert df.iloc[0]['Amount'] == pytest.approx(-1500.0)
    assert df.iloc[1]['Amount'] == pytest.approx(3200.5)


def test_header_not_on_first_row():
    raw = pd.DataFrame([
        ['FNB Business Account'],
        [''],
        ['Date', 'Transaction Description', 'Debit', 'Credit'],
        ['01/02/2024', 'Supplier ABC', '500.00', ''],
        ['02/02/2024', 'Customer payment', '', '1200.00'],
    ])
    df = normalize_bank_statement_dataframe(raw)
    assert len(df) == 2
    assert df.iloc[0]['Amount'] == pytest.approx(-500.0)
    assert df.iloc[1]['Amount'] == pytest.approx(1200.0)
    assert 'Supplier ABC' in df.iloc[0]['Description']


def test_signed_amount_column_with_currency():
    raw = pd.DataFrame([
        ['Date', 'Details', 'Amount'],
        ['2024-03-01', 'Bank charge', 'R -45.50'],
        ['2024-03-02', 'Deposit', 'R 1,000.00'],
    ])
    df = normalize_bank_statement_dataframe(raw)
    assert len(df) == 2
    assert df.iloc[0]['Amount'] == pytest.approx(-45.5)
    assert df.iloc[1]['Amount'] == pytest.approx(1000.0)


def test_empty_after_parse_raises():
    raw = pd.DataFrame([
        ['Date', 'Description', 'Amount'],
        ['not-a-date', 'Missing amount', ''],
    ])
    with pytest.raises(ValueError, match='No valid transaction rows'):
        normalize_bank_statement_dataframe(raw)
