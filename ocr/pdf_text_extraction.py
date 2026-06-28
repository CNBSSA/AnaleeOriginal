"""Tier-1 digital PDF bank-statement extraction (pypdf, no API call).

For text-bearing PDFs exported from SA online banking. Scanned/image PDFs
raise PdfStatementError so the caller can route to Claude Vision (Tier-2).
"""
from __future__ import annotations

import re
from decimal import Decimal
from typing import Optional

from .bank_profiles import detect_profile, parse_transaction_lines
from .statement_integrity import (
    ExtractionResult,
    StatementHeader,
    normalize_amount,
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


def extract_pdf_statement(
    file_bytes: bytes,
    opening_balance: Optional[Decimal] = None,
    closing_balance: Optional[Decimal] = None,
) -> ExtractionResult:
    """Extract from a digital PDF using per-bank layout profiles."""
    text = extract_text(file_bytes)
    profile = detect_profile(text)
    opening = opening_balance if opening_balance is not None else _find_balance(text, profile.opening_labels or _OPENING_LABELS)
    closing = closing_balance if closing_balance is not None else _find_balance(text, profile.closing_labels or _CLOSING_LABELS)
    if opening is None or closing is None:
        raise PdfStatementError(
            "could not locate opening and/or closing balance in the PDF text."
        )
    lines = parse_transaction_lines(text, profile)
    if not lines:
        raise PdfStatementError("no transaction rows found in PDF text.")
    return ExtractionResult(
        header=StatementHeader(
            bank=profile.display_name,
            opening_balance=opening,
            closing_balance=closing,
        ),
        lines=lines,
    )
