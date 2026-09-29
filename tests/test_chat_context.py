"""The chat assistant sees the financial year, not five lines (QA 2026-09-28)."""
from datetime import datetime, timedelta

from models import Account, CompanySettings, Transaction, User, db
from chat.routes import generate_ai_response, get_financial_context


def _books(app, n=12):
    with app.app_context():
        user = User(username='chatty', email='chatty@x.com', subscription_status='active')
        user.set_password('pw12345678')
        db.session.add(user)
        db.session.commit()
        db.session.add(CompanySettings(user_id=user.id, company_name='TEST', financial_year_end=2))
        sales = Account(link='i.100.000', name='Sales', category='Income',
                        sub_category='Income', user_id=user.id)
        rent = Account(link='e.400.000', name='Rent', category='Expenses',
                       sub_category='Expenses', user_id=user.id)
        db.session.add_all([sales, rent])
        db.session.flush()
        today = datetime.now()
        for i in range(n):
            db.session.add(Transaction(
                date=today - timedelta(days=7 * i), description=f'Line {i}',
                amount=1000.0 if i % 2 == 0 else -400.0,
                account_id=sales.id if i % 2 == 0 else rent.id, user_id=user.id))
        db.session.commit()
        return user.id


def test_context_covers_every_transaction_in_the_period_with_account_totals(canary_app):
    uid = _books(canary_app, n=12)
    with canary_app.app_context():
        ctx = get_financial_context(uid)
    assert ctx['total_transactions'] == 12
    assert ctx['lines_shown'] == 12
    assert ctx['period_label'] == 'financial year'
    totals = {r['account']: r for r in ctx['account_totals']}
    assert totals['Sales']['total'] == 6000.0 and totals['Sales']['count'] == 6
    assert totals['Rent']['total'] == -2400.0
    assert ctx['income'] == 6000.0 and ctx['expenses'] == 2400.0


def test_the_prompt_is_in_rand_and_names_the_period(canary_app, monkeypatch):
    uid = _books(canary_app, n=3)
    with canary_app.app_context():
        ctx = get_financial_context(uid)
    seen = {}

    class _Client:
        class messages:
            @staticmethod
            def create(**kw):
                seen['prompt'] = kw['messages'][0]['content']
                class R:
                    content = [type('T', (), {'text': 'ok'})()]
                return R()

    generate_ai_response(_Client(), 'How is rent looking?', ctx)
    prompt = seen['prompt']
    assert '$' not in prompt
    assert 'financial year' in prompt and 'Rent: R' in prompt
    assert 'Line 0' in prompt and 'Line 2' in prompt
