"""Batched suggestions: one AI call for many rows, both halves of the row.

Replaces one-Claude-call-per-transaction (each carrying the user's whole
~1 000-account chart) with one call per batch that returns the account AND the
explanation. These tests pin the safety rules, which matter more than the
speed-up:

* never invent an account — a name outside the chart is discarded, not
  fuzzy-matched onto something else;
* never guess when the AI is offline — return nothing so the caller applies
  nothing (the ASF rule from 57ee9a9, which the batch path never had);
* survive a truncated reply — salvage the rows that arrived;
* never overwrite what a person wrote, and always attribute machine writing.
"""
import json

import pytest

pytest.importorskip("pydantic")

from services.bulk_suggestions import (
    BULK_SUGGESTION_BATCH,
    RowSuggestion,
    suggest_for_rows,
)


class _Account:
    def __init__(self, id, name, category):
        self.id = id
        self.name = name
        self.category = category


CHART = [
    _Account(1, 'Bank Cheque Account 1', 'Assets'),
    _Account(2, 'Bank Charges', 'Expenses'),
    _Account(3, 'Sales', 'Income'),
]

ROWS = [
    {'index': 0, 'date': '2026-02-04', 'description': 'FNB FEE', 'amount': -59.0},
    {'index': 1, 'date': '2026-02-05', 'description': 'CLIENT PAYMENT', 'amount': 5000.0},
]


class _FakeClient:
    """Records the prompt and returns a canned reply."""

    def __init__(self, text):
        self.calls = 0
        self.last_prompt = None
        outer = self

        class _Messages:
            def create(self, **kwargs):
                outer.calls += 1
                outer.last_prompt = kwargs['messages'][0]['content']

                class _Block:
                    pass

                block = _Block()
                block.text = text

                class _Resp:
                    pass

                resp = _Resp()
                resp.content = [block]
                return resp

        self.messages = _Messages()


def _reply(items):
    return json.dumps(items)


def test_one_call_covers_the_whole_batch():
    """The point of the change: N rows cost ONE call, not N."""
    client = _FakeClient(_reply([
        {'i': 0, 'account': 'Bank Charges', 'confidence': 0.95, 'explanation': 'Monthly bank fee'},
        {'i': 1, 'account': 'Sales', 'confidence': 0.91, 'explanation': 'Payment from a client'},
    ]))
    out = suggest_for_rows(ROWS, CHART, client=client)

    assert client.calls == 1
    assert len(out) == 2
    assert out[0].account_name == 'Bank Charges'
    assert out[0].account_id == 2
    assert out[1].explanation == 'Payment from a client'


def test_the_chart_is_sent_once_per_batch_not_per_row():
    client = _FakeClient(_reply([{'i': 0, 'account': None, 'confidence': 0.1, 'explanation': 'x'}]))
    suggest_for_rows(ROWS, CHART, client=client)
    assert client.last_prompt.count('Bank Cheque Account 1') == 1


def test_an_account_outside_the_chart_is_discarded():
    """A hallucinated or renamed account must not be fuzzy-matched onto a real
    one — the explanation survives, the account does not."""
    client = _FakeClient(_reply([
        {'i': 0, 'account': 'Bank Charges Payable', 'confidence': 0.99,
         'explanation': 'Monthly bank fee'},
    ]))
    out = suggest_for_rows(ROWS, CHART, client=client)

    assert out[0].account_id is None
    assert out[0].confidence == 0.0, "an unmatched account must not carry confidence"
    assert out[0].explanation == 'Monthly bank fee'


def test_offline_never_falls_back_to_text_matching(monkeypatch):
    """No client -> no suggestions, so the caller applies nothing. The old
    per-row path fell through to a SequenceMatcher ratio whose score could
    clear the 0.85 auto-apply gate and post a guessed account."""
    monkeypatch.setattr('nlp_utils.get_claude_client', lambda: None)
    assert suggest_for_rows(ROWS, CHART) == {}


def test_an_api_failure_yields_nothing_rather_than_raising():
    class _Boom:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError('rate limited')

    assert suggest_for_rows(ROWS, CHART, client=_Boom()) == {}


def test_a_truncated_reply_salvages_the_complete_rows():
    full = _reply([
        {'i': 0, 'account': 'Bank Charges', 'confidence': 0.95, 'explanation': 'Monthly bank fee'},
        {'i': 1, 'account': 'Sales', 'confidence': 0.91, 'explanation': 'Payment from a client'},
    ])
    client = _FakeClient(full[:full.index('"Payment from a client"')])
    out = suggest_for_rows(ROWS, CHART, client=client)

    assert 0 in out and out[0].account_name == 'Bank Charges'


def test_a_hopeless_reply_yields_nothing():
    client = _FakeClient("I'm sorry, I can't help with that.")
    assert suggest_for_rows(ROWS, CHART, client=client) == {}


def test_confidence_is_clamped_and_unknown_rows_ignored():
    client = _FakeClient(_reply([
        {'i': 0, 'account': 'Bank Charges', 'confidence': 4.2, 'explanation': 'fee'},
        {'i': 99, 'account': 'Sales', 'confidence': 0.9, 'explanation': 'not our row'},
    ]))
    out = suggest_for_rows(ROWS, CHART, client=client)

    assert out[0].confidence == 1.0
    assert 99 not in out, "a row index we did not ask about must be ignored"


def test_markdown_fenced_reply_is_accepted():
    client = _FakeClient(
        "```json\n" + _reply([
            {'i': 0, 'account': 'Bank Charges', 'confidence': 0.9, 'explanation': 'fee'}]) + "\n```")
    out = suggest_for_rows(ROWS, CHART, client=client)
    assert out[0].account_name == 'Bank Charges'


def test_batch_size_is_bounded():
    assert 1 <= BULK_SUGGESTION_BATCH <= 50


def test_empty_inputs_are_safe():
    assert suggest_for_rows([], CHART, client=_FakeClient('[]')) == {}
    assert suggest_for_rows(ROWS, [], client=_FakeClient('[]')) == {}
