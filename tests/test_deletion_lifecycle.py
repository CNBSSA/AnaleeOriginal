"""Deleting a subscriber retires the person and KEEPS the books.

Ruling 2026-09-04 (docs/DATA_RETENTION.md). ``soft_delete()`` and
``restore_account()`` were called from admin/routes.py and forms/auth.py but
defined nowhere, so both paths raised AttributeError: admin deletion always
failed, and re-registering a previously-deleted address 500'd.

Implementing them was the easy half. The dangerous half was what sat behind
them: the admin handler went on to purge the user's transactions, accounts,
statement uploads and company settings, and every financial relationship on
User is cascade='all, delete-orphan'. Those are accounting records the customer
needs for their own filings and that the Companies Act s24(3), the Tax
Administration Act ss29-30 and the VAT Act s55 require be kept.

So the tests that matter here are the ones asserting records SURVIVE. If one of
them starts failing, the product is destroying customer books — do not "fix" it
by changing the assertion.
"""
import pytest

pytest.importorskip("flask_sqlalchemy")

from models import db, User, Account, Transaction


def _seed_user_with_books(app, email="owner@example.com", username="owner"):
    from datetime import datetime
    with app.app_context():
        user = User(username=username, email=email, subscription_status='active')
        user.set_password("Sup3rSecret!")
        db.session.add(user)
        db.session.flush()

        account = Account(link="1000", name="Bank", category="Current Assets",
                          user_id=user.id)
        db.session.add(account)
        db.session.flush()
        db.session.add(Transaction(
            date=datetime(2026, 3, 15), description="FEE IMMEDIATE PAYMENT",
            amount=-90.0, user_id=user.id, account_id=account.id))
        db.session.commit()
        return user.id


# ---- the lifecycle ---------------------------------------------------------

def test_soft_delete_blocks_login_but_keeps_every_record(app):
    uid = _seed_user_with_books(app)
    with app.app_context():
        user = db.session.get(User, uid)
        user.soft_delete()
        db.session.commit()

        user = db.session.get(User, uid)
        assert user is not None, "the row itself must survive"
        assert user.is_deleted is True
        assert user.is_active is False, "a deleted user must not be able to log in"

        assert Transaction.query.filter_by(user_id=uid).count() == 1, \
            "THE BOOKS WERE DESTROYED — see docs/DATA_RETENTION.md"
        assert Account.query.filter_by(user_id=uid).count() == 1, \
            "THE BOOKS WERE DESTROYED — see docs/DATA_RETENTION.md"


def test_restore_is_lossless_and_fail_closed(app):
    uid = _seed_user_with_books(app)
    with app.app_context():
        user = db.session.get(User, uid)
        user.soft_delete()
        db.session.commit()

        user = db.session.get(User, uid)
        user.restore_account()
        db.session.commit()

        user = db.session.get(User, uid)
        assert user.is_deleted is False
        # Fail-closed: restored to 'deactivated', NOT straight back to active.
        assert user.subscription_status == 'deactivated'
        # Lossless: the credential still works, so restore does not strand them.
        assert user.check_password("Sup3rSecret!")
        assert Transaction.query.filter_by(user_id=uid).count() == 1


def test_identifiers_are_only_released_for_a_deleted_account(app):
    uid = _seed_user_with_books(app)
    with app.app_context():
        user = db.session.get(User, uid)
        with pytest.raises(ValueError):
            user.release_identifiers_for_reregistration()

        user.soft_delete()
        user.release_identifiers_for_reregistration()
        db.session.commit()

        user = db.session.get(User, uid)
        assert user.email == f"deleted+{uid}@analee.invalid"
        assert user.username == f"deleted_user_{uid}"
        assert Transaction.query.filter_by(user_id=uid).count() == 1


# ---- the routes ------------------------------------------------------------

def test_re_registering_a_deleted_address_keeps_the_old_books(canary_app):
    """The path that used to db.session.delete(user) and cascade the books away."""
    EMAIL = "returning@example.com"
    client = canary_app.test_client()

    client.post("/auth/register", data={
        "username": "returning", "email": EMAIL,
        "password": "Sup3rSecret!", "confirm_password": "Sup3rSecret!",
    }, follow_redirects=True)
    client.get("/auth/logout")

    with canary_app.app_context():
        from datetime import datetime
        old = User.query.filter_by(email=EMAIL).first()
        old_id = old.id
        acct = Account(link="1000", name="Bank", category="Current Assets",
                       user_id=old_id)
        db.session.add(acct)
        db.session.flush()
        db.session.add(Transaction(date=datetime(2026, 3, 15), description="OLD BOOKS",
                                   amount=-90.0, user_id=old_id, account_id=acct.id))
        old.soft_delete()
        db.session.commit()

    resp = client.post("/auth/register", data={
        "username": "returning2", "email": EMAIL,
        "password": "An0therPass!", "confirm_password": "An0therPass!",
    }, follow_redirects=True)
    assert resp.status_code == 200

    with canary_app.app_context():
        new = User.query.filter_by(email=EMAIL).first()
        assert new is not None, "re-registration did not succeed"
        assert new.id != old_id, "the new signup reused the retired row"

        old = db.session.get(User, old_id)
        assert old is not None, "the old account was deleted"
        assert old.email == f"deleted+{old_id}@analee.invalid"
        assert Transaction.query.filter_by(user_id=old_id).count() == 1, \
            "RE-REGISTRATION DESTROYED THE PREVIOUS ACCOUNT'S BOOKS"


def test_a_live_address_still_cannot_be_re_registered(canary_app):
    """The guard that must not be loosened by the change above."""
    EMAIL = "live@example.com"
    client = canary_app.test_client()
    client.post("/auth/register", data={
        "username": "live", "email": EMAIL,
        "password": "Sup3rSecret!", "confirm_password": "Sup3rSecret!",
    }, follow_redirects=True)
    client.get("/auth/logout")

    client.post("/auth/register", data={
        "username": "impostor", "email": EMAIL,
        "password": "An0therPass!", "confirm_password": "An0therPass!",
    }, follow_redirects=True)

    with canary_app.app_context():
        assert User.query.filter_by(email=EMAIL).count() == 1
        user = User.query.filter_by(email=EMAIL).first()
        assert user.username == "live", "a live account was overwritten"
        assert user.check_password("Sup3rSecret!"), "a live password was replaced"


def test_the_admin_delete_handler_no_longer_purges_records():
    """Source-level guard. The purge is what made this dangerous."""
    import inspect
    import admin.routes as admin_routes

    src = inspect.getsource(admin_routes)
    start = src.find("deleting user {user.username}")
    assert start != -1
    end = src.find("def ", start)
    handler = src[start:end if end != -1 else len(src)]

    for destructive in ("Transaction.query.filter_by(user_id=user.id).delete()",
                        "Account.query.filter_by(user_id=user.id).delete()",
                        "BankStatementUpload.query.filter_by(user_id=user.id).delete()",
                        "CompanySettings.query.filter_by(user_id=user.id).delete()"):
        assert destructive not in handler, (
            f"the admin delete handler purges records again ({destructive}) — "
            "that destroys accounting records the Companies Act and SARS "
            "require be kept. See docs/DATA_RETENTION.md."
        )
