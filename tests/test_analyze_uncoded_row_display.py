"""Unit 1.5 — uncoded rows must not look coded on the Analyze screen.

TEST data only. The import stamps the statement bank on every row; the UI must
not present that stamp as the selected ledger account.
"""
import re
from datetime import datetime

import pytest

from models import Account, Transaction, UploadedFile, User, db
from services.analyze_processing import has_real_account_assignment


def _seed_analyze_rows(canary_app, *, bank_stamp=True, ledger_link='e.400.000'):
    """One bank-stamped row and one with a real ledger account."""
    with canary_app.app_context():
        user = User(username='uncoded', email='uncoded@x.test', subscription_status='active')
        user.set_password('pw12345!')
        db.session.add(user)
        db.session.commit()

        bank = Account(
            link='ca.810.001',
            name='TEST Bank Cheque Account 1',
            category='Assets',
            user_id=user.id,
            is_active=True,
        )
        ledger = Account(
            link=ledger_link,
            name='TEST Rent',
            category='Expenses',
            user_id=user.id,
            is_active=True,
        )
        uploaded = UploadedFile(filename='TEST-statement.csv', user_id=user.id)
        db.session.add_all([bank, ledger, uploaded])
        db.session.commit()

        stamped = Transaction(
            date=datetime(2026, 3, 1),
            description='TEST uncoded line',
            amount=-50.0,
            user_id=user.id,
            file_id=uploaded.id,
            account_id=bank.id if bank_stamp else None,
        )
        coded = Transaction(
            date=datetime(2026, 3, 2),
            description='TEST coded line',
            amount=-100.0,
            user_id=user.id,
            file_id=uploaded.id,
            account_id=ledger.id,
            explanation='TEST rent',
        )
        db.session.add_all([stamped, coded])
        db.session.commit()
        return user.id, uploaded.id, bank.id, ledger.id


def test_pipeline_and_ui_agree_on_bank_stamp_not_ledger(canary_app):
    """Rows the UI treats as unassigned must match has_real_account_assignment."""
    uid, fid, bank_id, ledger_id = _seed_analyze_rows(canary_app)
    with canary_app.app_context():
        stamped = Transaction.query.filter_by(file_id=fid, description='TEST uncoded line').one()
        coded = Transaction.query.filter_by(file_id=fid, description='TEST coded line').one()
        assert stamped.account_id == bank_id
        assert has_real_account_assignment(stamped) is False
        assert has_real_account_assignment(coded) is True
        assert coded.account_id == ledger_id


def test_analyze_page_bank_stamp_is_unassigned_in_markup(canary_app):
    from flask_login import login_user

    uid, fid, bank_id, _ledger_id = _seed_analyze_rows(canary_app)
    with canary_app.app_context():
        user = db.session.get(User, uid)
    from routes import analyze

    with canary_app.test_request_context(f'/analyze/{fid}'):
        login_user(user)
        html = analyze(fid)

    assert 'account-unassigned' in html
    assert '— Select account —' in html
    assert 'Bank: TEST Bank Cheque Account 1' in html
    assert f'value="{bank_id}"' in html
    # Bank option must not be pre-selected when only the import stamp is present.
    assert not re.search(
        rf'value="{bank_id}"[^>]*\sselected',
        html,
        flags=re.IGNORECASE,
    )


def test_analyze_page_ledger_row_still_shows_selected_account(canary_app):
    from flask_login import login_user

    uid, fid, _bank_id, ledger_id = _seed_analyze_rows(canary_app)
    with canary_app.app_context():
        user = db.session.get(User, uid)
    from routes import analyze

    with canary_app.test_request_context(f'/analyze/{fid}'):
        login_user(user)
        html = analyze(fid)

    assert re.search(
        rf'value="{ledger_id}"[^>]*\sselected',
        html,
        flags=re.IGNORECASE,
    )
    assert 'data-ledger-assigned="1"' in html


def test_analyze_template_wires_ledger_flag_not_raw_account_id():
    """Guard the Jinja rule: never select from account_id alone."""
    with open('templates/analyze.html', encoding='utf-8') as handle:
        markup = handle.read()

    assert 'has_ledger_account' in markup
    assert 'transaction_insights' in markup
    assert 'has_ledger_account and transaction.account_id' in markup.replace('\n', ' ')
