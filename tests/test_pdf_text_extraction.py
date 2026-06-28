"""Unit tests for Tier-1 digital PDF text extraction."""
import os
import sys
from decimal import Decimal

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.pdf_text_extraction import (  # noqa: E402
    PdfStatementError,
    extract_pdf_statement,
    parse_transaction_lines,
)


_SAMPLE_TEXT = """
FNB First National Bank
Account Statement

Opening Balance 1 000.00
15/03/2026 CARD PURCHASE CHECKERS 150.00- 850.00
20/03/2026 SALARY DEPOSIT 2 500.00 3 350.00
Closing Balance 3 350.00
"""


def test_parse_transaction_lines_generic_sa():
    lines = parse_transaction_lines(_SAMPLE_TEXT)
    assert len(lines) == 2
    assert lines[0].description == "CARD PURCHASE CHECKERS"
    assert lines[0].amount == Decimal("-150.00")
    assert lines[0].date == "2026-03-15"
    assert lines[1].amount == Decimal("2500.00")


def test_extract_pdf_statement_rejects_empty_bytes():
    with pytest.raises(PdfStatementError):
        extract_pdf_statement(b"")
