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


def test_headings_with_a_currency_qualifier_are_recognised():
    """QA re-test 2026-09-28: a statement headed 'Debit (R)' / 'Credit (R)'
    imported 0 rows because the heading matched no pattern."""
    import pandas as pd
    from bank_statements.format_detector import normalize_bank_statement_dataframe
    df = pd.DataFrame([
        ['Date', 'Description', 'Debit (R)', 'Credit (R)', 'Balance (R)'],
        ['2026-03-01', 'Sales', '', '15 000.00', '15000.00'],
        ['2026-03-02', 'Rent', '5 000.00', '', '10000.00'],
    ])
    out = normalize_bank_statement_dataframe(df)
    assert list(out['Amount']) == [15000.0, -5000.0]


def test_amount_heading_with_a_currency_qualifier_is_recognised():
    import pandas as pd
    from bank_statements.format_detector import normalize_bank_statement_dataframe
    df = pd.DataFrame([
        ['Transaction Date', 'Details', 'Amount (ZAR)'],
        ['2026-03-01', 'Sales', '150,00'],
    ])
    out = normalize_bank_statement_dataframe(df)
    assert list(out['Amount']) == [150.0]
