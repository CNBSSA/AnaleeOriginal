"""Generate AI Insights must describe the whole period, not one line.

QA re-test 2026-09-28: the narrative analysed 1 transaction out of 10 because
the page called the single-transaction method.
"""
import datetime
from unittest import mock

from models import db, User, CompanySettings, Transaction


def test_generate_insights_sends_every_transaction(canary_app):
    with canary_app.app_context():
        user = User(username='t', email='t@x.com', subscription_status='active')
        user.set_password('pw12345678')
        db.session.add(user)
        db.session.commit()
        uid = user.id
        db.session.add(CompanySettings(user_id=uid, company_name='TEST',
                                       financial_year_end=2))
        today = datetime.datetime.now()
        for i in range(10):
            db.session.add(Transaction(date=today - datetime.timedelta(days=i),
                                       description=f'Line {i}',
                                       amount=100.0 if i % 2 else -50.0,
                                       user_id=uid))
        db.session.commit()
    client = canary_app.test_client()
    client.post('/auth/login', data={'email': 't@x.com', 'password': 'pw12345678'})

    seen = {}

    def fake(self, transactions):
        seen['count'] = len(transactions)
        return {'success': True, 'insights': 'ok'}

    with mock.patch('ai_insights.FinancialInsightsGenerator.generate_insights', fake):
        client.post('/generate-insights')
    assert seen.get('count') == 10


def test_summary_is_in_rand_and_names_the_direction():
    from ai_insights import FinancialInsightsGenerator
    summary = FinancialInsightsGenerator._prepare_transaction_summary(
        None, [{'date': '2026-03-01', 'description': 'Rent', 'amount': -5000.0}])
    assert 'R5,000.00' in summary and '$' not in summary
    assert 'money out' in summary
