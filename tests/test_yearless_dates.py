"""Year-less transaction dates ("05 Sep") must resolve from the statement period.

Production failure (2026-07-18): a scanned SA statement printed line dates
without a year — every row was rejected ("unrecognised date format: '05 Sep'")
and the whole upload failed. The year lives in the statement period header;
resolve_date_with_period() recovers it, including Dec–Jan statements that span
a year boundary.
"""
import json
from datetime import date

import pytest

pytest.importorskip("pydantic")

from ocr.statement_integrity import resolve_date_with_period
from ocr.statement_extractor import _payload_to_result


def test_full_dates_pass_through_unchanged():
    assert resolve_date_with_period("2026-07-05") == "2026-07-05"
    assert resolve_date_with_period("05/07/2026") == "2026-07-05"


def test_yearless_resolves_inside_period():
    assert resolve_date_with_period(
        "05 Sep", period_start="2025-09-01", period_end="2025-09-30",
    ) == "2025-09-05"
    assert resolve_date_with_period(
        "05/09", period_start="2025-09-01", period_end="2025-09-30",
    ) == "2025-09-05"


def test_yearless_across_year_boundary():
    # Statement runs 15 Dec 2025 – 14 Jan 2026: December lines belong to 2025,
    # January lines to 2026.
    kw = dict(period_start="2025-12-15", period_end="2026-01-14")
    assert resolve_date_with_period("20 Dec", **kw) == "2025-12-20"
    assert resolve_date_with_period("05 Jan", **kw) == "2026-01-05"


def test_yearless_without_period_uses_most_recent_past():
    today = date(2026, 7, 18)
    assert resolve_date_with_period("05 Sep", today=today) == "2025-09-05"
    assert resolve_date_with_period("05 Jul", today=today) == "2026-07-05"


def test_garbage_still_raises():
    with pytest.raises(ValueError):
        resolve_date_with_period("not a date", period_start="2025-09-01")


def test_payload_with_yearless_lines_extracts():
    """End-to-end through _payload_to_result: the exact prod failure shape."""
    payload = {
        "bank": "Capitec",
        "period_start": "2025-09-01",
        "period_end": "2025-09-30",
        "opening_balance": "1000.00",
        "closing_balance": "700.00",
        "lines": [
            {"date": "05 Sep", "description": "CARD PURCHASE",
             "amount": "-100.00", "confidence": 0.95},
            {"date": "12 Sep", "description": "EFT PAYMENT",
             "amount": "-200.00", "confidence": 0.95},
        ],
    }
    result = _payload_to_result(payload)
    assert [ln.date for ln in result.lines] == ["2025-09-05", "2025-09-12"]
    assert len(result.lines) == 2
