"""Unit 2.3 — AI Insights grounded in bank statement lines (TEST data only)."""
import datetime
from unittest import mock

from models import CompanySettings, Transaction, User, db
from services.statement_insights import (
    analyze_statement_transactions,
    cash_flow_analysis_from_statement,
    format_statement_analysis_for_prompt,
)

PASSWORD = 'Sup3rSecret!'


def _sample_rows():
    return [
        {
            'date': '2026-03-01',
            'description': 'RENT OFFICE PARK',
            'amount': -5000.0,
            'category': 'Expenses',
            'account_name': 'Rent',
        },
        {
            'date': '2026-03-05',
            'description': 'RENT OFFICE PARK',
            'amount': -5000.0,
            'category': 'Expenses',
            'account_name': 'Rent',
        },
        {
            'date': '2026-03-10',
            'description': 'CLIENT DEPOSIT ACME',
            'amount': 15000.0,
            'category': 'Income',
            'account_name': 'Sales',
        },
        {
            'date': '2026-03-12',
            'description': 'SUPPLIER ONCE OFF 999',
            'amount': -45000.0,
            'category': 'Uncategorized',
            'account_name': 'Bank Cheque',
        },
        {
            'date': '2026-04-02',
            'description': 'SMALL FEE',
            'amount': -120.0,
            'category': 'Uncategorized',
            'account_name': 'Bank Cheque',
        },
    ]


def test_statement_analysis_reflects_spending_and_recurrence():
    analysis = analyze_statement_transactions(_sample_rows())
    assert analysis['empty'] is False
    assert analysis['totals']['count'] == 5
    assert analysis['totals']['inflow'] == 15000.0
    assert analysis['totals']['outflow'] == 55120.0

    labels = [item['label'] for item in analysis['top_spending']]
    assert 'Rent' in labels

    recurring_payees = {item['payee'] for item in analysis['recurring_payments']}
    assert 'rent office park' in recurring_payees

    assert analysis['unusual_amounts']
    assert any('45000' in str(item['amount']) for item in analysis['unusual_amounts'])

    assert len(analysis['monthly_cash_flow']) >= 2
    prompt = format_statement_analysis_for_prompt(analysis)
    assert 'Rent' in prompt or 'rent' in prompt.lower()
    assert 'R15,000.00' in prompt


def test_empty_statement_data_is_handled():
    empty = analyze_statement_transactions([])
    assert empty['empty'] is True
    assert empty['top_spending'] == []
    cash = cash_flow_analysis_from_statement([])
    assert 'No bank statement' in cash['current_status']


def test_prepare_summary_includes_statement_block():
    from ai_insights import FinancialInsightsGenerator

    summary = FinancialInsightsGenerator._prepare_transaction_summary(
        None, _sample_rows())
    assert 'Bank statement pattern analysis' in summary
    assert 'Top spending' in summary


def test_generate_insights_stores_statement_analysis_in_session(canary_app):
    with canary_app.app_context():
        user = User(username='ins', email='ins@example.com', subscription_status='active')
        user.set_password(PASSWORD)
        db.session.add(user)
        db.session.commit()
        db.session.add(CompanySettings(user_id=user.id, company_name='Co',
                                       financial_year_end=2))
        today = datetime.datetime(2026, 3, 15)
        for row in _sample_rows():
            db.session.add(Transaction(
                date=datetime.datetime.strptime(row['date'], '%Y-%m-%d'),
                description=row['description'],
                amount=row['amount'],
                user_id=user.id,
            ))
        db.session.commit()

    client = canary_app.test_client()
    client.post('/auth/login', data={'email': 'ins@example.com', 'password': PASSWORD})

    fake_analysis = analyze_statement_transactions(_sample_rows())

    def fake_generate(self, transactions):
        return {
            'success': True,
            'insights': 'TEST narrative',
            'analysis_type': 'ai_powered',
            'statement_analysis': fake_analysis,
        }

    with mock.patch(
            'ai_insights.FinancialInsightsGenerator.generate_insights', fake_generate):
        resp = client.post('/generate-insights')
    assert resp.status_code in (301, 302)

    with client.session_transaction() as sess:
        advice = sess.get('financial_advice') or {}
    assert advice.get('statement_analysis', {}).get('totals', {}).get('count') == 5
    assert advice.get('cash_flow_analysis', {}).get('current_status', '').startswith('Net cash flow: R')
