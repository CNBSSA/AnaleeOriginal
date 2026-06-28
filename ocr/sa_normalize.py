"""Shared SA amount/date normalisation for receipt and statement OCR."""
from __future__ import annotations

from decimal import Decimal
from typing import Optional

from .statement_integrity import normalize_amount, normalize_date


def parse_sa_amount(value, *, signed: bool = False) -> Optional[float]:
    """Parse a ZAR amount using the same rules as bank-statement integrity."""
    if value is None:
        return None
    try:
        number = float(normalize_amount(value))
        if not signed:
            number = abs(number)
        return round(number, 2)
    except (TypeError, ValueError):
        return None


def parse_sa_date(value) -> str:
    """Parse an SA date to ISO YYYY-MM-DD, or '' if unknown."""
    if not value:
        return ''
    try:
        return normalize_date(value)
    except ValueError:
        return ''
