"""Tier-1 digital PDF bank-statement extraction (pypdf, no API call).

For text-bearing PDFs exported from SA online banking. Scanned/image PDFs
raise PdfStatementError so the caller can route to Claude Vision (Tier-2).
"""
from __future__ import annotations

import re
from decimal import Decimal
from typing import Optional

from .statement_integrity import (
    ExtractionResult,
    StatementHeader,
    StatementLine,
    normalize_amount,
    normalize_date,
)

_DATE_RE = re.compile(
    r'^\s*('
    r'\d{4}[/-]\d{2}[/-]\d{2}'
    r'|\d{2}[/-]\d{2}[/-]\d{2,4}'
    r'|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}'
    r')\s+(.*)$'
)

_AMOUNT_RE = re.compile(r'\(?-?R?\s?[\d\s,]*\d\.\d{2}-?\)?')

_OPENING_LABELS = (
    'opening balance', 'balance brought forward', 'balance b/f', 'balance b/fwd',
    'opening bal', 'previous balance', 'bal b/f',
)
_CLOSING_LABELS = (
    'closing balance', 'balance carried forward', 'balance c/f', 'balance c/fwd',
    'closing bal', 'available balance', 'bal c/f',
)

# SA bank name hints for header detection (best-effort).
_SA_BANK_HINTS = (
    'fnb', 'first national bank', 'standard bank', 'absa', 'nedbank', 'capitec',
    'investec', 'african bank', 'tymebank', 'discovery bank', 'bidvest bank',
    'sasfin', 'grindrod', 'mercantile',
)


class PdfStatementError(Exception):
    """PDF cannot be read as a digital (text) statement."""


def extract_text(file_bytes: bytes) -> str:
    """Return all text from a digital PDF."""
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover
        from PyPDF2 import PdfReader  # type: ignore
    import io
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        raise PdfStatementError(f"could not read PDF: {exc}") from exc
    if not text.strip():
        raise PdfStatementError(
            "no extractable text — this looks like a scanned/image PDF."
        )
    return text


def detect_bank_name(text: str) -> Optional[str]:
    low = text.lower()
    for hint in _SA_BANK_HINTS:
        if hint in low:
            return hint.title()
    return None


def _find_balance(text: str, labels) -> Optional[Decimal]:
    for raw_line in text.splitlines():
        low = raw_line.lower()
        if any(lbl in low for lbl in labels):
            amounts = _AMOUNT_RE.findall(raw_line)
            if amounts:
                try:
                    return normalize_amount(amounts[-1])
                except ValueError:
                    continue
    return None


def parse_transaction_lines(text: str) -> list[StatementLine]:
    """Parse transaction rows from generic SA statement text layout."""
    lines: list[StatementLine] = []
    for raw_line in text.splitlines():
        m = _DATE_RE.match(raw_line)
        if not m:
            continue
        date_token, remainder = m.group(1), m.group(2)
        try:
            iso_date = normalize_date(date_token)
        except ValueError:
            continue
        amounts = _AMOUNT_RE.findall(remainder)
        if not amounts:
            continue
        if len(amounts) >= 2:
            amount_token, balance_token = amounts[-2], amounts[-1]
        else:
            amount_token, balance_token = amounts[-1], None
        try:
            amount = normalize_amount(amount_token)
        except ValueError:
            continue
        balance = None
        if balance_token is not None:
            try:
                balance = normalize_amount(balance_token)
            except ValueError:
                balance = None
        first_amt_pos = remainder.find(amounts[0])
        description = remainder[:first_amt_pos].strip() or remainder.strip()
        if not description:
            continue
        lines.append(StatementLine(
            date=iso_date, description=description, amount=amount, balance=balance,
        ))
    return lines


def extract_pdf_statement(
    file_bytes: bytes,
    opening_balance: Optional[Decimal] = None,
    closing_balance: Optional[Decimal] = None,
) -> ExtractionResult:
    """Extract from a digital PDF. Raises PdfStatementError on failure."""
    text = extract_text(file_bytes)
    opening = opening_balance if opening_balance is not None else _find_balance(text, _OPENING_LABELS)
    closing = closing_balance if closing_balance is not None else _find_balance(text, _CLOSING_LABELS)
    if opening is None or closing is None:
        raise PdfStatementError(
            "could not locate opening and/or closing balance in the PDF text."
        )
    lines = parse_transaction_lines(text)
    if not lines:
        raise PdfStatementError("no transaction rows found in PDF text.")
    return ExtractionResult(
        header=StatementHeader(
            bank=detect_bank_name(text),
            opening_balance=opening,
            closing_balance=closing,
        ),
        lines=lines,
    )
