"""Tier 0 automation unblock — the defects that stopped automation working.

Three independent failures, all verified against production behaviour before
being fixed:

1. ``/analyze/save-transaction`` and ``/analyze/replicate-explanation`` called
   ``dict.get(key, type=int)``. ``request.get_json()`` returns a plain dict and
   ``dict.get()`` takes no keyword arguments, so BOTH endpoints raised
   ``TypeError`` on every request and returned 500. The browser swallowed it
   (``console.error`` only), so an accountant changing the account dropdown saw
   no confirmation and lost the edit silently.

2. ``ReconciliationService`` never deletes, rewrites or commits anything — yet
   the UI reported "removed N duplicates, fixed N invalid dates". Removal stays
   un-automated on purpose (the duplicate rule groups on
   date+amount+description, so two genuine identical charges in one day are
   indistinguishable from a double capture); the reporting is what gets fixed.
"""
from datetime import datetime

import pytest

pytest.importorskip("flask_sqlalchemy")

from models import Account, Transaction, UploadedFile, User, db


def _seed(app):
    with app.app_context():
        user = User(username='tier0', email='tier0@x.test', subscription_status='active')
        user.set_password('pw12345!')
        db.session.add(user)
        db.session.commit()

        account = Account(link='ca.810.001', name='Bank Cheque Account 1',
                          category='Assets', user_id=user.id)
        uploaded = UploadedFile(filename='stmt.pdf', user_id=user.id,
                                upload_date=datetime.utcnow())
        db.session.add_all([account, uploaded])
        db.session.commit()

        txn = Transaction(date=datetime(2026, 2, 4), description='Magtape Credit',
                          amount=439.98, user_id=user.id, file_id=uploaded.id)
        db.session.add(txn)
        db.session.commit()
        return user.id, account.id, txn.id


def _payload(response):
    """Views may return a Response or a (Response, status) tuple."""
    if isinstance(response, tuple):
        return response[0].get_json(), response[1]
    return response.get_json(), response.status_code


def test_save_transaction_persists_instead_of_500(canary_app):
    """The account dropdown must actually save. This raised TypeError → 500."""
    user_id, account_id, txn_id = _seed(canary_app)

    from flask_login import login_user
    with canary_app.test_request_context(
            f'/analyze/save-transaction/{txn_id}', method='POST',
            json={'account_id': account_id, 'explanation': 'Medical aid refund'}):
        login_user(db.session.get(User, user_id))
        from routes import save_transaction
        body, status = _payload(save_transaction(txn_id))

    assert status == 200, f"expected 200, got {status}: {body}"
    assert body['success'] is True

    with canary_app.app_context():
        saved = db.session.get(Transaction, txn_id)
        assert saved.account_id == account_id, "the account change was not persisted"
        assert saved.explanation == 'Medical aid refund'


def test_save_transaction_rejects_a_non_numeric_account(canary_app):
    """Garbage must be a clean 400, not a 500 traceback."""
    user_id, _account_id, txn_id = _seed(canary_app)

    from flask_login import login_user
    with canary_app.test_request_context(
            f'/analyze/save-transaction/{txn_id}', method='POST',
            json={'account_id': 'not-a-number', 'explanation': ''}):
        login_user(db.session.get(User, user_id))
        from routes import save_transaction
        _body, status = _payload(save_transaction(txn_id))

    assert status == 400


def test_replicate_explanation_does_not_500(canary_app):
    """Recall — copy an explanation from a similar row — carried the same bug."""
    user_id, _account_id, txn_id = _seed(canary_app)

    with canary_app.app_context():
        target = db.session.get(Transaction, txn_id)
        source = Transaction(date=datetime(2026, 1, 4), description='Magtape Credit',
                             amount=439.98, user_id=user_id,
                             file_id=target.file_id,
                             explanation='Medical aid refund')
        db.session.add(source)
        db.session.commit()
        source_id = source.id

    from flask_login import login_user
    with canary_app.test_request_context(
            '/analyze/replicate-explanation', method='POST',
            json={'transaction_id': txn_id, 'similar_transaction_id': source_id}):
        login_user(db.session.get(User, user_id))
        from routes import replicate_explanation_api
        body, status = _payload(replicate_explanation_api())

    assert status == 200, f"got {status}: {body}"
    with canary_app.app_context():
        assert db.session.get(Transaction, txn_id).explanation == 'Medical aid refund'


def test_reconcile_reports_findings_not_fabricated_repairs(canary_app):
    """The service only inspects, so it must not claim it removed anything."""
    from bank_statements.reconciliation import ReconciliationService

    user_id, _account_id, _txn_id = _seed(canary_app)

    with canary_app.app_context():
        # Two identical rows: detectable as a possible duplicate.
        for _ in range(2):
            db.session.add(Transaction(
                date=datetime(2026, 3, 1), description='PARKING',
                amount=-50.0, user_id=user_id))
        db.session.commit()
        before = Transaction.query.filter_by(user_id=user_id).count()

        ok, result = ReconciliationService(user_id).perform_cleanup()
        stats = result['cleanup_stats']

        assert ok is True
        # Reports detection, using honest key names...
        assert 'duplicates_found' in stats
        assert 'invalid_dates_found' in stats
        assert stats['duplicates_found'] >= 1
        # ...and no key that claims a repair it never performed.
        assert 'duplicates_removed' not in stats
        assert 'invalid_dates_fixed' not in stats

        # Nothing was actually deleted — the books are untouched.
        assert Transaction.query.filter_by(user_id=user_id).count() == before


def test_reconcile_flash_never_claims_removal(canary_app):
    """Guard the wording itself: the route must not tell the user rows were
    removed or fixed when the service changes nothing."""
    import inspect

    from bank_statements import routes as bs_routes

    source = inspect.getsource(bs_routes.reconcile)
    lowered = source.lower()
    assert 'removed' not in lowered, "reconcile flash claims removal again"
    assert 'fixed' not in lowered, "reconcile flash claims repairs again"
    assert 'found' in lowered
