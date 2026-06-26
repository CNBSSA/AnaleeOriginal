"""
Money is stored and summed as Decimal (IFRS: exact to the cent).

Transaction.amount (and the goal amounts) are Numeric(18,2), so reads return
Decimal and sums tie out exactly — no binary-float drift like
0.1 + 0.2 + 0.3 == 0.6000000000000001. Boots the real app via canary_app.
"""
from decimal import Decimal

import pytest


def test_transaction_amount_is_decimal_and_sums_exactly(canary_app):
    app = canary_app
    from models import db, User, Account, Transaction
    from datetime import datetime

    with app.app_context():
        user = User(username="money", email="money@example.com")
        user.set_password("pw-123456")
        db.session.add(user)
        db.session.flush()
        account = Account(name="Bank", link="ca.1", category="Assets",
                          user_id=user.id, is_active=True)
        db.session.add(account)
        db.session.flush()

        # Classic float-drift values; assigned as floats (as the import path does).
        for amt in (0.1, 0.2, 0.3):
            db.session.add(Transaction(date=datetime(2026, 3, 1), description="x",
                                       amount=amt, user_id=user.id, account_id=account.id))
        db.session.commit()

        rows = Transaction.query.filter_by(user_id=user.id).all()
        # Stored/read back as Decimal, quantised to the cent.
        assert all(isinstance(t.amount, Decimal) for t in rows)
        total = sum(t.amount for t in rows)
        assert total == Decimal("0.60"), f"expected exact 0.60, got {total!r}"


def test_balanced_books_tie_out_to_zero(canary_app):
    app = canary_app
    from models import db, User, Account, Transaction
    from datetime import datetime

    with app.app_context():
        user = User(username="tie", email="tie@example.com")
        user.set_password("pw-123456")
        db.session.add(user)
        db.session.flush()
        account = Account(name="Bank", link="ca.1", category="Assets",
                          user_id=user.id, is_active=True)
        db.session.add(account)
        db.session.flush()

        # A debit and an equal-and-opposite credit must net to exactly zero.
        db.session.add(Transaction(date=datetime(2026, 3, 1), description="in",
                                   amount=100.10, user_id=user.id, account_id=account.id))
        db.session.add(Transaction(date=datetime(2026, 3, 2), description="out",
                                   amount=-100.10, user_id=user.id, account_id=account.id))
        db.session.commit()

        net = sum(t.amount for t in Transaction.query.filter_by(user_id=user.id))
        assert net == Decimal("0.00"), f"books did not tie out: {net!r}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
