"""
Unit tests for the bank-statement OCR service (no network).

Receipt OCR was removed (Analee is strictly cash-basis), so this covers the
statement wrapper (with a fake Claude client) and the pure duplicate-flagging
helper.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr import service  # noqa: E402


# --- fake Claude client ----------------------------------------------------
class _FakeBlock:
    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, text):
        self._text = text
        self.last_kwargs = None

    def stream(self, **kwargs):
        self.last_kwargs = kwargs
        resp = _FakeResponse(self._text)

        class _Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return resp

        return _Ctx()


class _FakeClient:
    def __init__(self, text):
        self.messages = _FakeMessages(text)


# --- statement wrapper -----------------------------------------------------
def test_extract_and_normalize_statement_signed_end_to_end():
    """Legacy wrapper delegates to the SA statement pipeline and keeps debit signs."""
    payload = {
        "opening_balance": "1000.00",
        "closing_balance": "970.00",
        "lines": [{
            "date": "01/03/2026",
            "description": "Card payment",
            "amount": "-30.00",
            "confidence": 0.7,
        }],
    }
    client = _FakeClient(json.dumps(payload))
    rows = service.extract_and_normalize_statement(b"%PDF fake", client=client)
    assert rows and rows[0]["amount"] == -30.0


def test_extract_and_normalize_statement_returns_empty_on_failure():
    """A client that yields no usable JSON -> [] (legacy emptiness contract)."""
    client = _FakeClient("sorry, no data here")
    assert service.extract_and_normalize_statement(b"%PDF fake", client=client) == []


# --- duplicate flagging ----------------------------------------------------
def test_mark_duplicates_flags_exact_match():
    rows = [
        {"date": "2026-03-01", "description": "Coffee Shop", "amount": -4.5},
        {"date": "2026-03-02", "description": "Salary", "amount": 2500.0},
    ]
    existing = [("2026-03-01", -4.5, "coffee shop")]  # already imported
    out = service.mark_duplicates(rows, existing)
    assert out[0]["duplicate"] is True
    assert out[1]["duplicate"] is False


def test_mark_duplicates_amount_or_date_mismatch_not_flagged():
    rows = [
        {"date": "2026-03-01", "description": "Coffee", "amount": -4.5},   # diff amount
        {"date": "2026-03-05", "description": "Coffee", "amount": -4.0},   # diff date
    ]
    existing = [("2026-03-01", -4.0, "coffee")]
    out = service.mark_duplicates(rows, existing)
    assert out[0]["duplicate"] is False
    assert out[1]["duplicate"] is False


def test_mark_duplicates_fuzzy_description_match():
    rows = [{"date": "2026-03-01", "description": "AMZN Mktp US*2X4", "amount": -19.99}]
    existing = [("2026-03-01", -19.99, "AMZN Mktp US*2X4 Amazon")]
    out = service.mark_duplicates(rows, existing)
    assert out[0]["duplicate"] is True


def test_mark_duplicates_blank_date_not_flagged():
    rows = [{"date": "", "description": "Coffee", "amount": -4.5}]
    existing = [("2026-03-01", -4.5, "coffee")]
    out = service.mark_duplicates(rows, existing)
    assert out[0]["duplicate"] is False


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
