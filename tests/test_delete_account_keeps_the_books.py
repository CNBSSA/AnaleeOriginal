"""Deleting a chart account must never erase bank lines posted to it.

Account.transactions is delete-orphan, so before 2026-09-28 one Delete click
on a used account deleted every transaction and statement posted to it.
"""
import datetime
import re

from models import db, User, Account, Transaction, HistoricalData


def _login(app):
    with app.app_context():
        user = User(username='t', email='t@x.com', subscription_status='active')
        user.set_password('pw12345678')
        db.session.add(user)
        db.session.commit()
        uid = user.id
    client = app.test_client()
    client.post('/auth/login', data={'email': 't@x.com', 'password': 'pw12345678'})
    return client, uid


def _account(uid, name):
    account = Account(link=f'x.{name}', name=name, category='Expenses',
                      sub_category='Other', user_id=uid)
    db.session.add(account)
    db.session.commit()
    return account.id


def test_an_account_with_posted_transactions_is_refused_and_the_lines_stay(canary_app):
    client, uid = _login(canary_app)
    with canary_app.app_context():
        aid = _account(uid, 'Rent')
        db.session.add(Transaction(date=datetime.datetime(2026, 3, 1),
                                   description='Rent', amount=-5000,
                                   account_id=aid, user_id=uid))
        db.session.commit()
    response = client.post(f'/account/{aid}/delete', follow_redirects=True)
    assert b'cannot be deleted while it holds 1 transaction.' in response.data
    with canary_app.app_context():
        assert db.session.get(Account, aid) is not None
        assert Transaction.query.filter_by(account_id=aid).count() == 1


def test_an_unused_account_still_deletes(canary_app):
    client, uid = _login(canary_app)
    with canary_app.app_context():
        aid = _account(uid, 'Spare')
        db.session.add(HistoricalData(date=datetime.datetime(2025, 3, 1),
                                      description='old', amount=1.0,
                                      account_id=aid, user_id=uid))
        db.session.commit()
    response = client.post(f'/account/{aid}/delete', follow_redirects=True)
    assert b'Account deleted successfully' in response.data
    with canary_app.app_context():
        assert db.session.get(Account, aid) is None
        assert HistoricalData.query.one().account_id is None
