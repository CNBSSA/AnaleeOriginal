"""Auto-detect SA bank statement column layouts (BooksXperts parity).

Bank exports rarely use exactly Date / Description / Amount on row 1.
This module finds the header row and maps Debit/Credit or Amount columns.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

_DATE_PATTERNS = {'date', 'transaction date', 'posting date', 'value date'}
_DESC_PATTERNS = {
    'description', 'description 1', 'transaction details', 'details',
    'narrative', 'particulars', 'trans description', 'transaction description',
    'reference', 'trans details',
}
_DEBIT_PATTERNS = {'debit', 'withdrawals', 'payment amount', 'debit amount', 'paid out'}
_CREDIT_PATTERNS = {'credit', 'deposits', 'receipt amount', 'credit amount', 'paid in'}
_AMOUNT_PATTERNS = {'amount', 'transaction amount', 'value', 'rand amount'}
_DESC2_PATTERNS = {'description 2', 'description 3', 'additional information'}


def _norm(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ''


def _find_index(headers_lower: list[str], patterns: set[str]) -> int | None:
    for index, header in enumerate(headers_lower):
        if header in patterns:
            return index
    return None


def _find_header_row(rows: list[list[Any]]) -> int | None:
    for index, row in enumerate(rows[:20]):
        headers_lower = [_norm(cell) for cell in row]
        has_date = _find_index(headers_lower, _DATE_PATTERNS) is not None
        has_amount = (
            _find_index(headers_lower, _DEBIT_PATTERNS) is not None
            or _find_index(headers_lower, _CREDIT_PATTERNS) is not None
            or _find_index(headers_lower, _AMOUNT_PATTERNS) is not None
        )
        if has_date and has_amount:
            return index
    return None


def _parse_number(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    cleaned = re.sub(r'[R\s,]', '', str(value).strip())
    if cleaned in ('', '-', 'nan', 'None'):
        return None
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _signed_amount(debit_raw: Any, credit_raw: Any, amount_raw: Any = None) -> float | None:
    debit = _parse_number(debit_raw)
    credit = _parse_number(credit_raw)
    if debit is not None or credit is not None:
        return (credit or 0.0) - (debit or 0.0)
    return _parse_number(amount_raw)


def normalize_bank_statement_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with Date, Description, Amount columns."""
    if df is None or df.empty:
        return pd.DataFrame(columns=['Date', 'Description', 'Amount'])

    raw_rows = df.fillna('').values.tolist()
    header_index = _find_header_row(raw_rows)
    if header_index is None:
        working = df.copy()
        working.columns = [str(col).strip() for col in working.columns]
    else:
        headers = [str(cell).strip() for cell in raw_rows[header_index]]
        data_rows = raw_rows[header_index + 1:]
        working = pd.DataFrame(data_rows, columns=headers)

    working.columns = [str(col).strip() for col in working.columns]
    headers_lower = [col.lower() for col in working.columns]

    date_col = _find_index(headers_lower, _DATE_PATTERNS)
    desc_col = _find_index(headers_lower, _DESC_PATTERNS)
    desc2_col = _find_index(headers_lower, _DESC2_PATTERNS)
    debit_col = _find_index(headers_lower, _DEBIT_PATTERNS)
    credit_col = _find_index(headers_lower, _CREDIT_PATTERNS)
    amount_col = (
        _find_index(headers_lower, _AMOUNT_PATTERNS)
        if debit_col is None and credit_col is None
        else None
    )

    if date_col is None:
        for col in working.columns:
            if col.lower() == 'date':
                date_col = working.columns.get_loc(col)
                break

    if desc_col is None:
        for col in working.columns:
            if col.lower() == 'description':
                desc_col = working.columns.get_loc(col)
                break

    if amount_col is None and 'amount' in headers_lower:
        amount_col = headers_lower.index('amount')

    if date_col is None or (desc_col is None and amount_col is None and debit_col is None and credit_col is None):
        raise ValueError(
            'Could not find required columns. Need Date plus Amount or Debit/Credit '
            '(or Description with Amount).'
        )

    normalized_rows: list[dict[str, Any]] = []
    for _, row in working.iterrows():
        date_val = row.iloc[date_col] if date_col is not None else None
        if pd.isna(date_val) or str(date_val).strip() == '':
            continue

        if desc_col is not None:
            description = str(row.iloc[desc_col]).strip()
            if desc2_col is not None:
                extra = str(row.iloc[desc2_col]).strip()
                if extra:
                    description = f'{description} {extra}'.strip()
        else:
            description = 'Bank transaction'

        debit_raw = row.iloc[debit_col] if debit_col is not None else None
        credit_raw = row.iloc[credit_col] if credit_col is not None else None
        amount_raw = row.iloc[amount_col] if amount_col is not None else None
        amount = _signed_amount(debit_raw, credit_raw, amount_raw)
        if amount is None:
            continue

        parsed_date = pd.to_datetime(date_val, errors='coerce', dayfirst=True)
        if pd.isna(parsed_date):
            continue

        if not description:
            description = 'Bank transaction'

        normalized_rows.append({
            'Date': parsed_date,
            'Description': description[:200],
            'Amount': amount,
        })

    result = pd.DataFrame(normalized_rows, columns=['Date', 'Description', 'Amount'])
    if result.empty:
        raise ValueError(
            'No valid transaction rows found. Check the file has Date and Amount '
            'or Debit/Credit columns with data below the header row.'
        )
    return result
