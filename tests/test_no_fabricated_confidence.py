"""Analee must never present an invented number as analysis.

Same rule as the ASF guard in 57ee9a9 ("refuse a guessed account instead of
showing one"), applied to the two places that still broke it:

1. ``/api/icountant/<id>/insights`` handed back the FIRST THREE ACCOUNTS IN THE
   CHART with an invented ``confidence: 0.5`` whenever the AI category did not
   map to an account. That fired for nearly every transaction — the category
   vocabulary is nlp_utils' personal-finance list (groceries, dining,
   personal_care, ...) while ``Account.category`` only ever holds Assets /
   Liabilities / Equity / Income / Expenses, so only "income" can match at all.
   The first three accounts of a seeded chart are Bank Cheque Account 1/2/3,
   and ``icountant.html`` auto-selects suggestion[0] into the account dropdown,
   so an accountant clicking through was steered into posting to a bank account.

2. ``/expense-forecast`` carried ``overall_confidence: 0.85`` and
   ``reliability_score: 0.80`` as hardcoded literals that nothing recomputed,
   rendered as "85.0%" / "80.0%" on the page and in the client-facing PDF.
"""
import inspect
import re
from datetime import datetime

import pytest

pytest.importorskip("flask_sqlalchemy")

from models import Account, Transaction, User, db


def _user_with_chart(app):
    """A user whose first accounts are the bank accounts — the real seed order."""
    with app.app_context():
        user = User(username='fab', email='fab@x.test', subscription_status='active')
        user.set_password('pw12345!')
        db.session.add(user)
        db.session.commit()

        db.session.add_all([
            Account(link='ca.810.001', name='Bank Cheque Account 1',
                    category='Assets', user_id=user.id),
            Account(link='ca.810.002', name='Bank Cheque Account 2',
                    category='Assets', user_id=user.id),
            Account(link='e.321.000', name='Bank Charges',
                    category='Expenses', user_id=user.id),
        ])
        txn = Transaction(date=datetime(2026, 2, 4), description='WOOLWORTHS SANDTON',
                          amount=-350.0, user_id=user.id)
        db.session.add(txn)
        db.session.commit()
        return user.id, txn.id


class _FakeInsights:
    """Mimics FinancialInsightsGenerator returning a personal-finance category
    that cannot map onto an SA chart of accounts — the common real case."""

    def __init__(self, category='groceries'):
        self._category = category

    def generate_transaction_insights(self, _rows):
        return {
            'insights': 'Looks like a supermarket purchase.',
            'category_suggestion': {
                'category': self._category,
                'confidence': 0.92,
                'explanation': 'Supermarket food purchase',
            },
        }


def _call_insights(app, user_id, txn_id, monkeypatch, category='groceries'):
    monkeypatch.setattr('routes.FinancialInsightsGenerator',
                        lambda *a, **k: _FakeInsights(category))
    from flask_login import login_user
    with app.test_request_context(f'/api/icountant/{txn_id}/insights'):
        login_user(db.session.get(User, user_id))
        from routes import icountant_transaction_insights
        resp = icountant_transaction_insights(txn_id)
    return (resp[0] if isinstance(resp, tuple) else resp).get_json()


def test_icountant_offers_nothing_rather_than_an_arbitrary_account(canary_app, monkeypatch):
    """An unmappable category must yield NO suggestions — not the first 3 accounts."""
    user_id, txn_id = _user_with_chart(canary_app)
    body = _call_insights(canary_app, user_id, txn_id, monkeypatch, category='groceries')

    assert body['success'] is True
    assert body['suggested_accounts'] == [], (
        "iCountant fabricated account suggestions again")
    assert 'suggestion_message' in body
    assert body['suggestion_message'], "the user must be told why there is no suggestion"


def test_icountant_never_emits_the_fabricated_half_confidence(canary_app, monkeypatch):
    """The invented 0.5 / 'Alternative suggestion' pair must not come back."""
    user_id, txn_id = _user_with_chart(canary_app)
    body = _call_insights(canary_app, user_id, txn_id, monkeypatch, category='personal_care')

    for suggestion in body['suggested_accounts']:
        assert suggestion.get('reason') != 'Alternative suggestion based on available accounts'
        assert suggestion.get('confidence') != 0.5


def test_icountant_still_returns_a_genuine_category_match(canary_app, monkeypatch):
    """The guard must not break the real path: a category that DOES map still
    produces suggestions drawn from that category."""
    user_id, txn_id = _user_with_chart(canary_app)
    body = _call_insights(canary_app, user_id, txn_id, monkeypatch, category='Expenses')

    names = [s['account_name'] for s in body['suggested_accounts']]
    assert names, "a mappable category must still yield suggestions"
    assert 'Bank Charges' in names
    assert not any(n.startswith('Bank Cheque Account') for n in names)


def _code_only(func):
    """Source of ``func`` with comment lines stripped.

    The fixes are documented in comments that necessarily quote the removed
    strings, so these guards must look at executable code, not commentary.
    """
    lines = inspect.getsource(func).splitlines()
    return '\n'.join(ln for ln in lines if not ln.strip().startswith('#'))


def test_icountant_source_has_no_arbitrary_fallback():
    """Source-level guard: restoring the accounts[:3] fallback fails the build."""
    import routes

    code = _code_only(routes.icountant_transaction_insights)
    assert 'Alternative suggestion based on available accounts' not in code
    assert "'confidence': 0.5" not in code
    # The genuine path slices `matching_accounts[:3]`; only the unqualified
    # `accounts[:3]` was the arbitrary "first three in the chart" fallback.
    assert not re.search(r'\bin\s+accounts\[:3\]', code)


def test_expense_forecast_has_no_hardcoded_confidence_literals():
    """The forecast must not assert a precision it never computed."""
    import routes

    code = _code_only(routes.expense_forecast)
    # The literals are gone from the payload...
    assert not re.search(r"'overall_confidence':\s*0\.85", code)
    assert not re.search(r"'reliability_score':\s*0\.80", code)
    # ...and the basis that IS measured is reported instead.
    assert 'months_observed' in code


def test_forecast_templates_do_not_render_the_removed_metrics():
    """Both the page and the client-facing PDF must stop printing them —
    a stale reference would also raise at render time."""
    for path in ('templates/expense_forecast.html',
                 'templates/pdf_templates/forecast_pdf.html'):
        with open(path, encoding='utf-8') as handle:
            markup = handle.read()
        assert 'overall_confidence' not in markup, f"{path} still prints a fabricated confidence"
        assert 'reliability_score' not in markup, f"{path} still prints a fabricated reliability"
        assert 'months_observed' in markup, f"{path} must state the basis it does have"
