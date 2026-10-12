"""QA D1 — cleanup of persisted R0.00 junk rows (TEST data only)."""
from datetime import datetime

from models import Transaction, UploadedFile, User, db
from services.junk_transaction_cleanup import (
    cleanup_definite_junk_transactions,
    find_definite_junk_transactions,
    transaction_is_definite_junk,
)


def test_transaction_is_definite_junk_matches_qa_descriptions():
    assert transaction_is_definite_junk("Schd Trxn No Av Bal Pmt To Payroll", 0.0)
    assert transaction_is_definite_junk("Schd Trxn No Av Bal Pmt To Festus", 0.0)
    assert not transaction_is_definite_junk("EFT to supplier", 0.0)


def test_cleanup_dry_run_leaves_rows(app):
    with app.app_context():
        user = User(username="junkdry", email="junkdry@x.test", subscription_status="active")
        db.session.add(user)
        db.session.flush()
        f = UploadedFile(filename="TEST.pdf", user_id=user.id, upload_date=datetime.utcnow())
        db.session.add(f)
        db.session.flush()
        db.session.add(
            Transaction(
                date=datetime(2026, 3, 15),
                description="Schd Trxn No Av Bal Pmt To Payroll",
                amount=0.0,
                user_id=user.id,
                file_id=f.id,
            )
        )
        db.session.commit()

        report = cleanup_definite_junk_transactions(db.session, apply=False)
        assert report.matched_junk == 1
        assert report.deleted == 0
        assert Transaction.query.filter_by(user_id=user.id).count() == 1


def test_cleanup_apply_removes_junk_and_keeps_real_rows(app):
    with app.app_context():
        user = User(username="junkapply", email="junkapply@x.test", subscription_status="active")
        db.session.add(user)
        db.session.flush()
        f = UploadedFile(filename="TEST.pdf", user_id=user.id, upload_date=datetime.utcnow())
        db.session.add(f)
        db.session.flush()
        db.session.add_all(
            [
                Transaction(
                    date=datetime(2026, 3, 15),
                    description="Schd Trxn No Av Bal Pmt To Festus",
                    amount=0.0,
                    user_id=user.id,
                    file_id=f.id,
                ),
                Transaction(
                    date=datetime(2026, 3, 16),
                    description="CARD PURCHASE",
                    amount=-50.0,
                    user_id=user.id,
                    file_id=f.id,
                ),
                Transaction(
                    date=datetime(2026, 3, 17),
                    description="Fee waiver pending",
                    amount=0.0,
                    user_id=user.id,
                    file_id=f.id,
                ),
            ]
        )
        db.session.commit()

        junk = find_definite_junk_transactions(db.session, user_ids=[user.id])
        assert len(junk) == 1

        report = cleanup_definite_junk_transactions(
            db.session, user_ids=[user.id], apply=True
        )
        assert report.deleted == 1
        remaining = Transaction.query.filter_by(user_id=user.id).order_by(Transaction.date).all()
        assert len(remaining) == 2
        assert remaining[0].description == "CARD PURCHASE"
        assert remaining[1].description == "Fee waiver pending"
