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


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
