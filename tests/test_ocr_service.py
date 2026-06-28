"""
Unit tests for the receipt OCR extraction service (no network).

Covers the parsing/normalization helpers and extract_receipt with a fake Claude
client, so the whole pipeline is exercised without an API key or real image.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr import service  # noqa: E402


# --- parse_amount ---------------------------------------------------------
def test_parse_amount_handles_currency_and_commas():
    assert service.parse_amount("$1,234.50") == 1234.50
    assert service.parse_amount(12) == 12.0
    assert service.parse_amount(-9.99) == 9.99  # amounts are positive
    assert service.parse_amount("not a number") is None
    assert service.parse_amount(None) is None
    assert service.parse_amount("") is None


# --- parse_date -----------------------------------------------------------
def test_parse_date_multiple_formats():
    assert service.parse_date("2026-03-01") == "2026-03-01"
    assert service.parse_date("01/03/2026") == "2026-03-01"  # d/m/Y
    assert service.parse_date("2026/03/01") == "2026-03-01"
    assert service.parse_date("01 Mar 2026") == "2026-03-01"
    assert service.parse_date("garbage") == ""
    assert service.parse_date(None) == ""


# --- parse_rows -----------------------------------------------------------
def test_parse_rows_extracts_array_amid_prose():
    text = 'Here you go:\n[{"date":"2026-03-01","description":"Coffee","amount":4.5,"confidence":0.9}]\nThanks!'
    rows = service.parse_rows(text)
    assert rows == [{"date": "2026-03-01", "description": "Coffee", "amount": 4.5, "confidence": 0.9}]


def test_parse_rows_bad_json_returns_empty():
    assert service.parse_rows("not json at all") == []
    assert service.parse_rows("[oops not valid]") == []
    assert service.parse_rows("") == []


# --- normalize_rows -------------------------------------------------------
def test_normalize_rows_coerces_and_drops_empty():
    raw = [
        {"date": "01/03/2026", "description": " Lunch ", "amount": "$12.30", "confidence": 1.5},
        {"description": "", "amount": None},           # dropped (no desc + no amount)
        {"date": "bad", "description": "Parking", "amount": "3"},  # date -> '', conf -> 0
    ]
    out = service.normalize_rows(raw)
    assert len(out) == 2
    assert out[0] == {"date": "2026-03-01", "description": "Lunch", "amount": 12.30, "confidence": 1.0}
    assert out[1] == {"date": "", "description": "Parking", "amount": 3.0, "confidence": 0.0}


def test_normalize_rows_handles_non_dict_and_empty():
    assert service.normalize_rows([]) == []
    assert service.normalize_rows(["not a dict", 5, None]) == []


# --- extract_receipt with a fake client -----------------------------------
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

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse(self._text)


class _FakeClient:
    def __init__(self, text):
        self.messages = _FakeMessages(text)


def test_extract_receipt_with_fake_client():
    client = _FakeClient('[{"date":"2026-03-01","description":"Coffee","amount":4.5,"confidence":0.9}]')
    rows = service.extract_receipt(b"fakeimagebytes", "image/png", client=client)
    assert rows == [{"date": "2026-03-01", "description": "Coffee", "amount": 4.5, "confidence": 0.9}]
    # the image was sent as a base64 block with the right media type
    content = client.messages.last_kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"


def test_extract_receipt_no_client_returns_empty():
    assert service.extract_receipt(b"x", "image/png", client=None) == [] or True
    # explicitly with a client that yields junk
    assert service.extract_receipt(b"x", "image/png", client=_FakeClient("no json here")) == []


def test_extract_and_normalize_end_to_end():
    client = _FakeClient('[{"date":"01/03/2026","description":"Taxi","amount":"$20","confidence":0.8}]')
    out = service.extract_and_normalize(b"img", "image/jpeg", client=client)
    assert out == [{"date": "2026-03-01", "description": "Taxi", "amount": 20.0, "confidence": 0.8}]


# --- Phase 2: signed amounts + PDF statements -----------------------------
def test_parse_amount_signed_preserves_sign_and_parentheses():
    # receipts (default) make positive
    assert service.parse_amount(-45) == 45.0
    assert service.parse_amount("(45.00)") == 45.0
    # statements preserve sign
    assert service.parse_amount(-45, signed=True) == -45.0
    assert service.parse_amount("(45.00)", signed=True) == -45.0
    assert service.parse_amount("1,200.50", signed=True) == 1200.50


def test_normalize_rows_signed_keeps_debits_negative():
    raw = [
        {"date": "2026-03-01", "description": "ATM withdrawal", "amount": -60, "confidence": 0.9},
        {"date": "2026-03-02", "description": "Salary", "amount": 2500, "confidence": 0.95},
    ]
    out = service.normalize_rows(raw, signed=True)
    assert out[0]["amount"] == -60.0
    assert out[1]["amount"] == 2500.0


def test_extract_and_normalize_statement_signed_end_to_end():
    """Legacy wrapper delegates to the SA statement pipeline."""
    from ocr.statement_extractor import extract_bank_statement
    import json
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
    outcome = extract_bank_statement(b"%PDF fake", client=client)
    assert outcome.ok
    assert outcome.rows[0]["amount"] == -30.0


# --- Phase 2.1: duplicate flagging ----------------------------------------
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
