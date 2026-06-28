"""Unit tests for SA bank-statement integrity gate."""
import os
import sys
from decimal import Decimal

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.statement_integrity import (  # noqa: E402
    ExtractionResult,
    StatementHeader,
    StatementLine,
    normalize_amount,
    normalize_date,
    self_audit,
)


def test_normalize_amount_sa_formats():
    assert normalize_amount("R 1,234.56") == Decimal("1234.56")
    assert normalize_amount("100.00-") == Decimal("-100.00")
    assert normalize_amount("(50.00)") == Decimal("-50.00")
    assert normalize_amount("25.00 Dr") == Decimal("-25.00")
    assert normalize_amount("25.00 Cr") == Decimal("25.00")


def test_normalize_date_sa_day_first():
    assert normalize_date("15/03/2026") == "2026-03-15"
    assert normalize_date("01 Jan 2026") == "2026-01-01"


def test_self_audit_reconciled():
    result = ExtractionResult(
        header=StatementHeader(opening_balance="1000.00", closing_balance="850.00"),
        lines=[
            StatementLine(date="2026-03-01", description="ATM", amount="-150.00"),
        ],
    )
    card = self_audit(result)
    assert card.reconciled is True
    assert card.variance == Decimal("0.00")


def test_self_audit_variance_detected():
    result = ExtractionResult(
        header=StatementHeader(opening_balance="1000.00", closing_balance="900.00"),
        lines=[
            StatementLine(date="2026-03-01", description="ATM", amount="-150.00"),
        ],
    )
    card = self_audit(result)
    assert card.reconciled is False
    assert card.variance == Decimal("50.00")
