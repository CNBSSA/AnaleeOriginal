"""Auto-detect SA bank statement column layouts (BooksXperts parity).

Bank exports rarely use exactly Date / Description / Amount on row 1.
This module finds the header row and maps Debit/Credit or Amount columns.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd
from utils.date_parsing import parse_statement_date

_DATE_PATTERNS = {
    'date', 'transaction date', 'posting date', 'value date', 'trans date',
    'date posted', 'processing date', 'effective date',
}
_DESC_PATTERNS = {
    'description', 'description 1', 'transaction details', 'details',
    'narrative', 'particulars', 'trans description', 'transaction description',
    'reference', 'trans details',
}
# Capitec exports "Money In" / "Money Out" (+ a separate "Fee" column);
# Nedbank and others use "Debits"/"Credits" or "Withdrawal"/"Deposit". Before
# 2026-09-28 none of those were recognised, so a real Capitec CSV was refused
# with "Could not find required columns".
_DEBIT_PATTERNS = {
    'debit', 'debits', 'withdrawal', 'withdrawals', 'payment amount',
    'debit amount', 'paid out', 'money out', 'amount out',
}
_CREDIT_PATTERNS = {
    'credit', 'credits', 'deposit', 'deposits', 'receipt amount',
    'credit amount', 'paid in', 'money in', 'amount in',
}
_FEE_PATTERNS = {'fee', 'fees', 'fee amount', 'bank fee', 'bank fees'}
_AMOUNT_PATTERNS = {'amount', 'transaction amount', 'value', 'rand amount'}
_DESC2_PATTERNS = {'description 2', 'description 3', 'additional information'}


_HEADER_QUALIFIER = re.compile(r'\s*[\(\[][^\)\]]*[\)\]]\s*$')


def _norm(value: Any) -> str:
    """A heading as the patterns spell it: lower case, and without a trailing
    qualifier in brackets, so ``Debit (R)``, ``Credit [ZAR]`` and
    ``Amount (R)`` read as debit / credit / amount. Before 2026-09-28 a
    statement headed ``Debit (R)`` / ``Credit (R)`` matched nothing and
    imported 0 rows."""
    if value is None:
        return ''
    text = ' '.join(str(value).replace('\u00a0', ' ').split()).lower()
    text = _HEADER_QUALIFIER.sub('', text).rstrip(':*').strip()
    return text


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


_DR_CR_SUFFIX = re.compile(r'\s*(?<![A-Za-z])(DR|CR)\.?\s*$', re.IGNORECASE)


def _parse_number(value: Any) -> float | None:
    """Read an amount the way SA bank exports actually write one.

    Handles, in addition to plain ``-150.00``: a currency prefix (``R``,
    ``ZAR``), spaces or commas as thousands separators (``R 1 150.00``,
    ``1,150.00``), a decimal comma (``150,00`` — semicolon-delimited exports),
    negatives in brackets (``(1,150.00)``), a trailing minus (``150.00-``, FNB)
    and a ``Dr``/``Cr`` suffix. Returns None when the cell is not a number.
    Before 2026-09-28 a bracketed amount returned None and its row was dropped
    without a word, and ``150,00`` was read as 15 000.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).replace('\u00a0', ' ').strip()
    if text in ('', '-', 'nan', 'NaN', 'None'):
        return None

    negative = False
    suffix = _DR_CR_SUFFIX.search(text)
    if suffix:
        negative = suffix.group(1).upper() == 'DR'
        text = text[:suffix.start()].strip()
    if text.startswith('(') and text.endswith(')'):
        negative = True
        text = text[1:-1].strip()
    if text.endswith('-'):
        negative = True
        text = text[:-1].strip()

    text = re.sub(r'(?i)zar', '', text)
    text = re.sub(r'[Rr\s]', '', text)
    if text.startswith('+'):
        text = text[1:]
    if text.startswith('-'):
        negative = not negative
        text = text[1:]
    if not text:
        return None

    if ',' in text and '.' in text:
        # Whichever separator comes last is the decimal point.
        if text.rfind(',') > text.rfind('.'):
            text = text.replace('.', '').replace(',', '.')
        else:
            text = text.replace(',', '')
    elif ',' in text:
        # One comma followed by one or two digits is a decimal comma
        # (150,00); anything else is a thousands separator (15,000).
        if text.count(',') == 1 and re.search(r',\d{1,2}$', text):
            text = text.replace(',', '.')
        else:
            text = text.replace(',', '')

    if not re.fullmatch(r'\d+(\.\d+)?|\.\d+', text):
        return None
    number = float(text)
    return -number if negative else number


def _signed_amount(debit_raw: Any, credit_raw: Any, amount_raw: Any = None,
                   fee_raw: Any = None) -> float | None:
    """Money in is positive, money out (and a separate fee) negative.

    Debit/credit columns are read by magnitude: Capitec writes "Money Out" as
    ``-150.00`` and other banks as ``150.00``; both mean 150 left the account.
    (Subtracting a negative debit used to turn a payment into a receipt.)
    """
    debit = _parse_number(debit_raw)
    credit = _parse_number(credit_raw)
    fee = _parse_number(fee_raw)
    if debit is not None or credit is not None or fee is not None:
        if debit is None and credit is None and amount_raw is not None:
            base = _parse_number(amount_raw)
            if base is None:
                return -abs(fee)
            return base - abs(fee)
        return abs(credit or 0.0) - abs(debit or 0.0) - abs(fee or 0.0)
    return _parse_number(amount_raw)


def find_header_row(rows: list[list[Any]]) -> int | None:
    """Public alias used by the CSV reader to choose a delimiter."""
    return _find_header_row(rows)


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
    headers_lower = [_norm(col) for col in working.columns]

    date_col = _find_index(headers_lower, _DATE_PATTERNS)
    desc_col = _find_index(headers_lower, _DESC_PATTERNS)
    desc2_col = _find_index(headers_lower, _DESC2_PATTERNS)
    debit_col = _find_index(headers_lower, _DEBIT_PATTERNS)
    credit_col = _find_index(headers_lower, _CREDIT_PATTERNS)
    fee_col = _find_index(headers_lower, _FEE_PATTERNS)
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
        if header_index is None:
            first = next((r for r in raw_rows if any(str(c).strip() for c in r)), [])
            seen = [str(c).strip() for c in first if str(c).strip()][:12]
        else:
            seen = [str(c) for c in working.columns if str(c).strip()][:12]
        raise ValueError(
            'Could not find the column headings. Analee needs a Date column and '
            'an Amount column (or Debit and Credit, or Money In and Money Out). '
            'Column headings found: ' + (', '.join(str(c) for c in seen) or 'none')
            + '.'
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
        fee_raw = row.iloc[fee_col] if fee_col is not None else None
        amount = _signed_amount(debit_raw, credit_raw, amount_raw, fee_raw)
        if amount is None:
            continue

        # Day-first for DD/MM, as written for ISO. dayfirst=True alone read
        # '2025-04-03' as 4 March, and excel_reader hands every cell over as
        # a string (dtype=str), so real Excel date cells came through here.
        parsed_date = parse_statement_date(date_val)
        if parsed_date is None:
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
