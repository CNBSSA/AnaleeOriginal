"""Unit tests for the SA bank-statement extraction orchestrator."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.statement_extractor import (  # noqa: E402
    BankStatementExtraction,
    _payload_to_result,
    extract_bank_statement,
)


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


_SA_PAYLOAD = {
    "bank": "FNB",
    "account_number": "****1234",
    "opening_balance": "1000.00",
    "closing_balance": "850.00",
    "lines": [
        {
            "date": "15/03/2026",
            "description": "CARD PURCHASE CHECKERS",
            "amount": "-150.00",
            "balance": "850.00",
            "confidence": 0.95,
        },
    ],
}


def test_payload_to_result_builds_extraction():
    result = _payload_to_result(_SA_PAYLOAD)
    assert result.header.bank == "FNB"
    assert len(result.lines) == 1
    assert str(result.lines[0].amount) == "-150.00"


def test_extract_bank_statement_claude_tier_with_fake_client():
    client = _FakeClient(json.dumps(_SA_PAYLOAD))
    # Non-text PDF bytes force Tier-1 to fail → Tier-2 Claude
    outcome = extract_bank_statement(b"%PDF-1.4 not real text", client=client)
    assert isinstance(outcome, BankStatementExtraction)
    assert outcome.ok
    assert outcome.method == "claude_vision"
    assert len(outcome.rows) == 1
    assert outcome.report_card is not None
    assert outcome.report_card.reconciled is True
    assert client.messages.last_kwargs["model"]  # model id passed through


def test_extract_bank_statement_ai_unavailable():
    outcome = extract_bank_statement(b"%PDF fake", client=None)
    # No ANTHROPIC_API_KEY in test env → Tier-2 fails after Tier-1 fails
    assert not outcome.ok
    assert outcome.error_code in ("AI_UNAVAILABLE", "EXTRACTION_FAILED")
