"""Work order #33: money in is never given an Expenses account, and money out is
never given an Income account, by the AI suggestion.

The rule used to live only as a sentence in the prompt; nothing checked the
answer, so a +15,000 receipt came back "Salaries" at 0.95 and cleared the 0.85
auto-apply gate. The guard drops the account (the line stays open for a person)
and keeps the explanation. Assets, Liabilities and Equity go either way
(transfers, loans, owner money) and are never blocked.

Scoped re-open granted by the Chairman, 2026-10-07 ("Reopen Analee").
"""
import json

import pytest

pytest.importorskip("pydantic")

from services.bulk_suggestions import suggest_for_rows


class _Account:
    def __init__(self, id, name, category):
        self.id, self.name, self.category = id, name, category


CHART = [
    _Account(1, 'Bank Cheque Account 1', 'Assets'),
    _Account(2, 'Salaries', 'Expenses'),
    _Account(3, 'Sales', 'Income'),
    _Account(4, 'Loan from Director', 'Liabilities'),
    _Account(5, "Owner's Contribution", 'Equity'),
]


class _Client:
    def __init__(self, items):
        text = json.dumps(items)

        class _Messages:
            def create(self, **kwargs):
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


def _suggest(amount, account, explanation="why"):
    rows = [{'index': 0, 'date': '2026-02-05', 'description': 'LINE', 'amount': amount}]
    reply = [{"i": 0, "account": account, "confidence": 0.95, "explanation": explanation}]
    return suggest_for_rows(rows, CHART, client=_Client(reply))[0]


def test_money_in_is_never_given_an_expenses_account():
    s = _suggest(15000.0, 'Salaries')
    assert s.account_id is None
    assert s.confidence == 0.0


def test_money_out_is_never_given_an_income_account():
    s = _suggest(-9000.0, 'Sales')
    assert s.account_id is None
    assert s.confidence == 0.0


def test_the_explanation_survives_a_dropped_account():
    s = _suggest(15000.0, 'Salaries', explanation='Client payment received')
    assert s.explanation == 'Client payment received'


def test_money_in_to_income_is_kept():
    s = _suggest(15000.0, 'Sales')
    assert s.account_id == 3
    assert s.confidence == 0.95


def test_money_out_to_expenses_is_kept():
    s = _suggest(-9000.0, 'Salaries')
    assert s.account_id == 2
    assert s.confidence == 0.95


@pytest.mark.parametrize('amount', [5000.0, -5000.0])
@pytest.mark.parametrize('account,account_id', [
    ('Bank Cheque Account 1', 1), ('Loan from Director', 4), ("Owner's Contribution", 5)])
def test_balance_sheet_accounts_go_either_way(amount, account, account_id):
    s = _suggest(amount, account)
    assert s.account_id == account_id


def test_category_is_matched_ignoring_case():
    chart = [_Account(9, 'Wages', 'EXPENSES')]
    rows = [{'index': 0, 'description': 'X', 'amount': 100.0}]
    reply = [{"i": 0, "account": "Wages", "confidence": 0.9, "explanation": "e"}]
    assert suggest_for_rows(rows, chart, client=_Client(reply))[0].account_id is None
