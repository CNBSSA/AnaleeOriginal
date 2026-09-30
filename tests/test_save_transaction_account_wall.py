"""save_transaction must not store another user's account (2026-09-30).

Found in the Analee simplification read-through and verified by hand: the route
scoped the TRANSACTION to the signed-in user but stored any posted
``account_id`` as sent, so a crafted request could point a user's own line at
an account in someone else's chart — the same "the filter scopes the row, not
the account" class closed in BooksXperts (#818). Every other path that sets an
account (CSV/PDF import, fan-out, Ask Analee, historical data) already refused
one that is not the user's. TEST data only.
"""
import pytest

pytest.importorskip("flask_sqlalchemy")

from models import Account, Transaction, User, db
from test_tier0_automation_unblock import _payload, _seed


def _other_users_account(app):
    with app.app_context():
        other = User(username='wall-other', email='wall-other@x.test',
                     subscription_status='active')
        other.set_password('pw12345!')
        db.session.add(other)
        db.session.commit()
        acc = Account(link='ex.100.000', name='TEST Other Rent', category='Expenses',
                      user_id=other.id)
        db.session.add(acc)
        db.session.commit()
        return acc.id


def _save(app, user_id, txn_id, account_id):
    from flask_login import login_user
    with app.test_request_context(f'/analyze/save-transaction/{txn_id}', method='POST',
                                  json={'account_id': account_id, 'explanation': 'TEST'}):
        login_user(db.session.get(User, user_id))
        from routes import save_transaction
        return _payload(save_transaction(txn_id))


def test_another_users_account_is_refused_and_nothing_changes(canary_app):
    user_id, _own, txn_id = _seed(canary_app)
    foreign = _other_users_account(canary_app)
    with canary_app.app_context():
        before = db.session.get(Transaction, txn_id).account_id
    body, status = _save(canary_app, user_id, txn_id, foreign)
    assert status == 400
    assert 'not in your chart' in body['error']
    with canary_app.app_context():
        saved = db.session.get(Transaction, txn_id)
        assert saved.account_id == before
        assert saved.account_id != foreign
        assert saved.explanation != 'TEST'


def test_a_missing_account_id_is_refused(canary_app):
    user_id, _own, txn_id = _seed(canary_app)
    _body, status = _save(canary_app, user_id, txn_id, 999999)
    assert status == 400


def test_the_users_own_account_still_saves(canary_app):
    user_id, own, txn_id = _seed(canary_app)
    _body, status = _save(canary_app, user_id, txn_id, own)
    assert status == 200
    with canary_app.app_context():
        assert db.session.get(Transaction, txn_id).account_id == own
