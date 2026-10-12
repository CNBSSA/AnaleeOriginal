"""Festus 2026-09-30: "approve 1, 2, 5" (Analee).

1. Ask Analee also lists lines still showing only the bank account. Both import
   paths put the statement's bank on every line, so a queue of account-less
   lines never showed an imported statement.
2. PDF import offers — and accepts — only bank-type accounts (bank, cash,
   credit card / overdraft); a statement can only come from one of those.
5. Send TB carries the financial year on screen into the link it hands THE
   ACCOUNTANTS (Copy share link already did).
TEST data only. Frozen engine and TB core only called.
"""
from datetime import date, datetime
from unittest import mock

import pytest

pytest.importorskip("flask_sqlalchemy")

from test_asf_degradation_guard import _register_and_login
from test_practice_send_tb_club import API, _FakeResponse, _club_workspace_client, _enable


def _setup_chart(app, email):
    from models import Account, User, db
    with app.app_context():
        u = User.query.filter_by(email=email).one()
        bank = Account(link='ca.810.001', name='TEST Bank Cheque', category='Assets', user_id=u.id)
        rent = Account(link='e.400.000', name='TEST Rent', category='Expenses', user_id=u.id)
        db.session.add_all([bank, rent])
        db.session.commit()
        return u.id, bank.id, rent.id


def _line(app, uid, account_id, desc):
    from models import Transaction, UploadedFile, db
    with app.app_context():
        f = UploadedFile(filename='TEST.csv', user_id=uid, upload_date=datetime.utcnow())
        db.session.add(f)
        db.session.flush()
        db.session.add(Transaction(date=datetime(2026, 3, 1), description=desc, amount=-50.0,
                                   user_id=uid, account_id=account_id, file_id=f.id))
        db.session.commit()


# ---- 1 -----------------------------------------------------------------------

def test_ask_analee_lists_a_line_still_on_the_bank(canary_app):
    from test_asf_degradation_guard import EMAIL
    client = canary_app.test_client()
    _register_and_login(client)
    uid, bank, rent = _setup_chart(canary_app, EMAIL)
    _line(canary_app, uid, bank, 'TEST still on the bank')
    _line(canary_app, uid, rent, 'TEST already decided')
    page = client.get('/icountant').get_data(as_text=True)
    assert 'TEST still on the bank' in page
    assert 'Nothing is waiting here' not in page


def test_ask_analee_is_empty_when_every_line_has_a_real_account(canary_app):
    from test_asf_degradation_guard import EMAIL
    client = canary_app.test_client()
    _register_and_login(client)
    uid, bank, rent = _setup_chart(canary_app, EMAIL)
    _line(canary_app, uid, rent, 'TEST already decided')
    page = client.get('/icountant').get_data(as_text=True)
    assert 'Nothing is waiting here' in page


# ---- 2 -----------------------------------------------------------------------

def _ocr_app():
    from test_ocr_invalid_dates import _login, _make_app, _seed_user
    app = _make_app()
    uid = _seed_user(app)
    client = app.test_client()
    _login(client, uid)
    return app, uid, client


def _accounts(app, uid):
    from models import Account, db
    with app.app_context():
        bank = Account(link='cl.810.001', name='TEST Credit Card', category='Liabilities', user_id=uid)
        rent = Account(link='e.400.000', name='TEST Rent', category='Expenses', user_id=uid)
        db.session.add_all([bank, rent])
        db.session.commit()
        return bank.id, rent.id


def _confirm(client, account_id):
    return client.post('/ocr/statement/confirm', data={
        'date': ['2026-03-15'], 'description': ['TEST'], 'amount': ['10.00'],
        'filename': 'TEST.pdf', 'account_id': str(account_id)}, follow_redirects=False)


def test_pdf_import_refuses_a_non_bank_account_and_imports_nothing():
    from models import Transaction
    app, uid, client = _ocr_app()
    _bank, rent = _accounts(app, uid)
    r = _confirm(client, rent)
    assert r.status_code == 302 and '/analyze/' not in r.headers['Location']
    with app.app_context():
        assert Transaction.query.filter_by(user_id=uid).count() == 0


def test_pdf_import_accepts_a_bank_type_account():
    from models import Transaction
    app, uid, client = _ocr_app()
    bank, _rent = _accounts(app, uid)
    r = _confirm(client, bank)
    assert '/analyze/' in r.headers['Location']
    with app.app_context():
        assert Transaction.query.filter_by(user_id=uid).one().account_id == bank


def test_pdf_account_list_is_bank_accounts_only():
    import ocr.routes as ocr_routes
    app, uid, client = _ocr_app()
    bank, rent = _accounts(app, uid)
    from flask_login import login_user
    from models import User, db
    with app.test_request_context('/'):
        login_user(db.session.get(User, uid))
        assert [a.id for a in ocr_routes._user_accounts()] == [bank]


# ---- 5 -----------------------------------------------------------------------

def test_send_tb_carries_the_year_on_screen(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch)
    _enable(monkeypatch)
    today = date.today().isoformat()
    fake = _FakeResponse(body={"received": True})
    with mock.patch("practice_layer.requests.post", return_value=fake) as post:
        client.post("/practice/send-tb", data={"as_at": today}, follow_redirects=True)
    assert post.call_args.args[0] == API + "/tb-drop"
    assert f"as_at={today}" in post.call_args.kwargs["json"]["share_url"]


def test_send_tb_without_a_year_is_unchanged(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch)
    _enable(monkeypatch)
    with mock.patch("practice_layer.requests.post",
                    return_value=_FakeResponse(body={"received": True})) as post:
        client.post("/practice/send-tb", follow_redirects=True)
    url = post.call_args.kwargs["json"]["share_url"]
    assert "as_at=" not in url and "financial_year=" not in url


def test_the_send_tb_form_keeps_the_year_from_the_page():
    import os
    base = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'templates/base.html'), encoding='utf-8').read()
    assert base.count("for key in ('as_at', 'financial_year')") == 2
    assert base.count('Send TB &rarr; THE ACCOUNTANTS') == 2  # the button itself is unchanged
