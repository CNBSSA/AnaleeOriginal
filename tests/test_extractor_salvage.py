"""Extraction must survive imperfect Claude replies.

Production failure (2026-07-18): a 5-page scanned statement died with
"Could not read transactions from that PDF (ValueError)" — the model reply
was truncated by the max_tokens ceiling, and one bad header balance could
fail an otherwise perfect extraction. These tests pin the three defenses:
truncated-JSON salvage, tolerant header balances, and the raised reply budget.
"""
import json
from decimal import Decimal

import pytest

pytest.importorskip("pydantic")

from ocr.statement_extractor import (
    _claude_extract_pdf_payload,
    _parse_claude_json,
    _payload_to_result,
    _salvage_truncated_payload,
)


def _line(i):
    return {
        "date": f"2026-07-{i:02d}",
        "description": f"PAYMENT {i}",
        "amount": f"-{i}00.00",
        "balance": None,
        "confidence": 0.95,
    }


def _payload_json(n_lines):
    return json.dumps({
        "bank": "FNB",
        "account_number": "****1234",
        "period_start": "2026-07-01",
        "period_end": "2026-07-31",
        "opening_balance": "1000.00",
        "closing_balance": "400.00",
        "lines": [_line(i + 1) for i in range(n_lines)],
    })


def test_parse_intact_json_still_works():
    payload = _parse_claude_json(_payload_json(3))
    assert payload["bank"] == "FNB"
    assert len(payload["lines"]) == 3


def test_truncated_reply_salvages_complete_lines():
    """Cut the reply mid-way through the 4th line object — the salvage parser
    must recover the header and the 3 complete lines instead of raising."""
    full = _payload_json(4)
    cut = full.find('"PAYMENT 4"')  # truncate inside the 4th object
    payload = _parse_claude_json(full[:cut])
    assert payload["bank"] == "FNB"
    assert payload["opening_balance"] == "1000.00"
    assert [ln["description"] for ln in payload["lines"]] == [
        "PAYMENT 1", "PAYMENT 2", "PAYMENT 3"]


def test_salvage_handles_braces_inside_strings():
    text = ('{"bank": "Weird {Bank}", "lines": ['
            '{"date": "2026-07-01", "description": "REF {A1}", '
            '"amount": "-10.00", "confidence": 0.9}, {"date": "2026-07-02", "descr')
    payload = _salvage_truncated_payload(text)
    assert payload["bank"] == "Weird {Bank}"
    assert len(payload["lines"]) == 1
    assert payload["lines"][0]["description"] == "REF {A1}"


def test_hopeless_reply_still_raises():
    with pytest.raises(ValueError):
        _parse_claude_json("I'm sorry, I cannot read this document.")


def test_unreadable_header_balance_does_not_fail_extraction():
    payload = json.loads(_payload_json(2))
    payload["opening_balance"] = "N/A"
    payload["closing_balance"] = "see page 2"
    result = _payload_to_result(payload)
    assert result.header.opening_balance is None
    assert result.header.closing_balance is None
    assert len(result.lines) == 2  # lines still captured -> "unverified" path


def test_readable_balances_still_parse():
    result = _payload_to_result(json.loads(_payload_json(1)))
    assert result.header.opening_balance == Decimal("1000.00")
    assert result.header.closing_balance == Decimal("400.00")


class _FakeMessages:
    def __init__(self, text, stop_reason="end_turn"):
        self._text = text
        self._stop = stop_reason
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs

        class _Block:
            pass

        block = _Block()
        block.text = self._text

        class _Resp:
            pass

        resp = _Resp()
        resp.content = [block]
        resp.stop_reason = self._stop
        return resp


class _FakeClient:
    def __init__(self, text, stop_reason="end_turn"):
        self.messages = _FakeMessages(text, stop_reason)


def test_claude_call_uses_raised_reply_budget():
    client = _FakeClient(_payload_json(1))
    payload = _claude_extract_pdf_payload(b"%PDF-fake", "prompt", client)
    assert payload["bank"] == "FNB"
    assert client.messages.last_kwargs["max_tokens"] == 32000


def test_truncated_reply_with_max_tokens_stop_reason_salvages():
    full = _payload_json(3)
    client = _FakeClient(full[:full.find('"PAYMENT 3"')], stop_reason="max_tokens")
    payload = _claude_extract_pdf_payload(b"%PDF-fake", "prompt", client)
    assert len(payload["lines"]) == 2
