"""BX-AN-1 / Unit 2.1: categorizer prompt carries amount, direction, and history.

The model used to see description-only context in practice (amount and sign were
easy to miss). The bulk suggester now labels money IN vs money OUT explicitly,
includes prior payee practice when available, and keeps the work-order #33
direction guard as a safety net when the model still errs.

These tests use TEST data and a fake client only — no live API calls.
"""
import json

import pytest

pytest.importorskip("pydantic")

from services.bulk_suggestions import suggest_for_rows
from services.history_matching import HistoryEntry


class _Account:
    def __init__(self, id, name, category):
        self.id, self.name, self.category = id, name, category


CHART = [
    _Account(1, 'Bank Cheque Account 1', 'Assets'),
    _Account(2, 'Salaries', 'Expenses'),
    _Account(3, 'Sales', 'Income'),
    _Account(4, 'Bank Charges', 'Expenses'),
    _Account(5, 'Bank Savings Account', 'Assets'),
    _Account(6, 'Loan from Director', 'Liabilities'),
]


class _RecordingClient:
    """Returns a fixed JSON reply and records the prompt."""

    def __init__(self, items):
        self.last_prompt = None
        text = json.dumps(items)
        outer = self

        class _Messages:
            def create(self_outer, **kwargs):
                outer.last_prompt = kwargs['messages'][0]['content']

                class _Block:
                    pass

                block = _Block()
                block.text = text

                class _Resp:
                    pass

                return type('R', (), {'content': [block]})()

        self.messages = _Messages()


def _one_row(description, amount):
    return [{
        'index': 0,
        'date': '2026-02-05',
        'description': description,
        'amount': amount,
    }]


# --- Prompt context (what the model is told) ---------------------------------

def test_prompt_labels_money_in_and_money_out_explicitly():
    rows = [
        {'index': 0, 'date': '2026-02-04', 'description': 'FEE', 'amount': -59.0},
        {'index': 1, 'date': '2026-02-05', 'description': 'CLIENT PAYMENT', 'amount': 15000.0},
    ]
    client = _RecordingClient([
        {'i': 0, 'account': 'Bank Charges', 'confidence': 0.9, 'explanation': 'fee'},
        {'i': 1, 'account': 'Sales', 'confidence': 0.9, 'explanation': 'sale'},
    ])
    suggest_for_rows(rows, CHART, client=client)

    assert 'money OUT (payment/debit)' in client.last_prompt
    assert 'money IN (receipt/credit)' in client.last_prompt
    assert 'amount=15000.0 ZAR' in client.last_prompt
    assert 'amount=-59.0 ZAR' in client.last_prompt
    assert 'Respect direction' in client.last_prompt or 'money IN' in client.last_prompt


def test_prompt_includes_relevant_payee_history():
    rows = _one_row('Magtape Credit Medihelp Smh0363383 20210204', -1200.0)
    history = {
        'magtape credit medihelp': HistoryEntry(
            key='magtape credit medihelp',
            account_counts={4: 3},
            account_names={4: 'Bank Charges'},
            explanation='Medical aid debit order',
            total=3,
        ),
    }
    client = _RecordingClient([
        {'i': 0, 'account': 'Bank Charges', 'confidence': 0.88, 'explanation': 'medical'},
    ])
    suggest_for_rows(rows, CHART, client=client, history_index=history)

    assert 'magtape credit medihelp' in client.last_prompt
    assert 'Bank Charges' in client.last_prompt
    assert 'Medical aid debit order' in client.last_prompt


# --- Labeled examples (guard + sensible categories) --------------------------

@pytest.mark.parametrize('description,amount,good_account,bad_account', [
    ('CLIENT PAYMENT ACME PTY', 15000.0, 'Sales', 'Salaries'),
    ('SALARY BATCH FEB', -9000.0, 'Salaries', 'Sales'),
    ('FNB SERVICE FEE', -59.0, 'Bank Charges', 'Sales'),
])
def test_money_in_must_not_stick_to_expense_nor_money_out_to_income(
        description, amount, good_account, bad_account):
    """Wrong-direction model answers are dropped; correct ones are kept."""
    rows = _one_row(description, amount)
    wrong = suggest_for_rows(
        rows, CHART,
        client=_RecordingClient([
            {'i': 0, 'account': bad_account, 'confidence': 0.95, 'explanation': 'x'},
        ]),
    )[0]
    assert wrong.account_id is None
    assert wrong.confidence == 0.0

    right = suggest_for_rows(
        rows, CHART,
        client=_RecordingClient([
            {'i': 0, 'account': good_account, 'confidence': 0.92, 'explanation': 'ok'},
        ]),
    )[0]
    assert right.account_name == good_account
    assert right.confidence == 0.92


@pytest.mark.parametrize('amount,account,account_id', [
    (15000.0, 'Bank Savings Account', 5),
    (-10000.0, 'Bank Savings Account', 5),
    (5000.0, 'Loan from Director', 6),
    (-5000.0, 'Loan from Director', 6),
])
def test_direction_ambiguous_balance_sheet_lines_still_accept_suggestions(
        amount, account, account_id):
    """Transfers and loan movements are not blocked by the income/expense guard."""
    rows = _one_row('TRANSFER BETWEEN OWN ACCOUNTS', amount)
    out = suggest_for_rows(
        rows, CHART,
        client=_RecordingClient([
            {'i': 0, 'account': account, 'confidence': 0.9, 'explanation': 'transfer'},
        ]),
    )[0]
    assert out.account_id == account_id
