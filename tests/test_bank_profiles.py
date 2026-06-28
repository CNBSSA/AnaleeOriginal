"""Unit tests for per-bank layout profiles."""
import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.bank_profiles import (  # noqa: E402
    CAPITEC,
    FNB,
    detect_profile,
    parse_transaction_lines,
)


CAPITEC_TEXT = """
Capitec Bank
Account Statement
Opening Balance 1 000.00
15/03/2026 SHOPRITE  0.00  150.00  850.00
20/03/2026 SALARY    2 500.00  0.00  3 350.00
Closing Balance 3 350.00
"""

FNB_TEXT = """
FNB First National Bank
15/03/2026 CARD PURCHASE CHECKERS 150.00- 850.00
"""


def test_detect_profile_capitec():
    assert detect_profile(CAPITEC_TEXT).profile_id == "capitec"


def test_detect_profile_fnb():
    assert detect_profile(FNB_TEXT).profile_id == "fnb"


def test_capitec_money_in_out_columns():
    lines = parse_transaction_lines(CAPITEC_TEXT, CAPITEC)
    assert len(lines) == 2
    assert lines[0].amount == Decimal("-150.00")
    assert lines[1].amount == Decimal("2500.00")


def test_fnb_trailing_minus():
    lines = parse_transaction_lines(FNB_TEXT, FNB)
    assert len(lines) == 1
    assert lines[0].amount == Decimal("-150.00")
