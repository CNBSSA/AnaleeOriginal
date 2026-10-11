"""Unit 2.2 — AI Assistant Chat sees the current statement (read-only)."""
from datetime import datetime

from models import Account, CompanySettings, Transaction, UploadedFile, User, db
from chat.routes import generate_ai_response, get_financial_context
def _user_with_statement(app, *, stmt_name='TEST_stmt_march.csv', line_desc='TEST chat line rent'):
    with app.app_context():
        user = User(username='chatuser', email='chatuser@example.com', subscription_status='active')
        user.set_password('pw12345678')
        db.session.add(user)
        db.session.commit()
        db.session.add(CompanySettings(user_id=user.id, company_name='TEST Co', financial_year_end=2))
        bank = Account(
            link='ca.810.001', name='Bank Cheque Account 1', category='Assets',
            sub_category='Assets', user_id=user.id)
        rent = Account(
            link='e.400.001', name='Rent', category='Expenses',
            sub_category='Expenses', user_id=user.id)
        db.session.add_all([bank, rent])
        db.session.flush()
        older = UploadedFile(filename='older.csv', user_id=user.id, bank_account_id=bank.id)
        current = UploadedFile(filename=stmt_name, user_id=user.id, bank_account_id=bank.id)
        db.session.add_all([older, current])
        db.session.flush()
        db.session.add(Transaction(
            date=datetime(2026, 1, 10), description='OLD file line',
            amount=-100.0, user_id=user.id, file_id=older.id, account_id=bank.id))
        db.session.add(Transaction(
            date=datetime(2026, 2, 5), description=line_desc,
            amount=-2500.0, user_id=user.id, file_id=current.id, account_id=bank.id,
            explanation='Office rent'))
        db.session.add(Transaction(
            date=datetime(2026, 2, 6), description='TEST coded sale',
            amount=9000.0, user_id=user.id, file_id=current.id, account_id=rent.id,
            explanation='Misfiled for test'))
        db.session.commit()
        return user.id, current.id, older.id


def test_context_includes_current_statement_lines_and_categorisation(canary_app):
    uid, file_id, _older = _user_with_statement(canary_app)
    with canary_app.app_context():
        ctx = get_financial_context(uid)
    assert ctx['statement'] is not None
    assert ctx['statement']['filename'] == 'TEST_stmt_march.csv'
    assert ctx['statement']['transaction_count'] == 2
    assert ctx['statement']['unprocessed_count'] == 1
    descriptions = {t['description'] for t in ctx['statement']['transactions']}
    assert 'TEST chat line rent' in descriptions
    rent_row = next(t for t in ctx['statement']['transactions'] if t['description'] == 'TEST chat line rent')
    assert rent_row['status'] == 'needs account'
    assert rent_row['has_explanation'] is True


def test_explicit_file_id_selects_that_statement(canary_app):
    uid, current_id, older_id = _user_with_statement(canary_app)
    with canary_app.app_context():
        ctx = get_financial_context(uid, file_id=older_id)
    assert ctx['statement']['file_id'] == older_id
    assert ctx['statement']['transaction_count'] == 1
    assert ctx['statement']['transactions'][0]['description'] == 'OLD file line'


def test_prompt_mentions_statement_line_and_balances(canary_app, monkeypatch):
    uid, _fid, _ = _user_with_statement(canary_app)
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

    generate_ai_response(_Client(), 'What is on my March statement?', ctx)
    prompt = seen['prompt']
    assert 'TEST chat line rent' in prompt
    assert 'Current bank statement' in prompt
    assert 'Trial balance account balances' in prompt
    assert 'needs account' in prompt


def test_empty_workspace_has_no_statement_block(canary_app):
    with canary_app.app_context():
        user = User(username='empty', email='empty@example.com', subscription_status='active')
        user.set_password('pw12345678')
        db.session.add(user)
        db.session.commit()
        uid = user.id
        ctx = get_financial_context(uid)
    assert ctx['statement'] is None
    assert ctx['total_transactions'] == 0

    class _Client:
        class messages:
            @staticmethod
            def create(**kw):
                class R:
                    content = [type('T', (), {'text': 'ok'})()]
                return R()

    text = generate_ai_response(_Client(), 'Anything there?', ctx)
    assert text == 'ok'


def test_prompt_handles_missing_statement_gracefully(canary_app):
    with canary_app.app_context():
        user = User(username='empty2', email='empty2@example.com', subscription_status='active')
        user.set_password('pw12345678')
        db.session.add(user)
        db.session.commit()
        ctx = get_financial_context(user.id)
    seen = {}

    class _Client:
        class messages:
            @staticmethod
            def create(**kw):
                seen['prompt'] = kw['messages'][0]['content']
                class R:
                    content = [type('T', (), {'text': 'ok'})()]
                return R()

    generate_ai_response(_Client(), 'Show my statement', ctx)
    assert 'none uploaded yet' in seen['prompt'].lower()


def test_fully_categorised_line_counts_complete(canary_app):
    with canary_app.app_context():
        user = User(username='done', email='done@example.com', subscription_status='active')
        user.set_password('pw12345678')
        db.session.add(user)
        db.session.commit()
        db.session.add(CompanySettings(user_id=user.id, company_name='TEST', financial_year_end=2))
        bank = Account(link='ca.810.002', name='Bank', category='Assets',
                       sub_category='Assets', user_id=user.id)
        sales = Account(link='i.100.001', name='Sales', category='Income',
                        sub_category='Income', user_id=user.id)
        db.session.add_all([bank, sales])
        db.session.flush()
        f = UploadedFile(filename='one.csv', user_id=user.id, bank_account_id=bank.id)
        db.session.add(f)
        db.session.flush()
        db.session.add(Transaction(
            date=datetime(2026, 3, 1), description='TEST complete line',
            amount=500.0, user_id=user.id, file_id=f.id, account_id=sales.id,
            explanation='Sale'))
        db.session.commit()
        ctx = get_financial_context(user.id)
    assert ctx['statement']['categorization']['complete'] == 1
    assert ctx['statement']['unprocessed_count'] == 0
    row = ctx['statement']['transactions'][0]
    assert row['status'] == 'complete'
