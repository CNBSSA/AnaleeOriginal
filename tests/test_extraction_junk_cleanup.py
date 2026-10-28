"""Cleanup for zero-amount extraction junk already stored as transactions (QA D1)."""
from datetime import datetime

from models import Transaction, UploadedFile, User, db
from services.extraction_junk_cleanup import cleanup_junk_transactions, find_junk_transaction_ids


def test_find_and_remove_persisted_junk_rows(app, sample_user):
    user_id = sample_user
    with app.app_context():
        uf = UploadedFile(
            filename="old.pdf",
            user_id=user_id,
            upload_date=datetime.utcnow(),
        )
        db.session.add(uf)
        db.session.flush()

        junk_a = Transaction(
            date=datetime(2026, 3, 1),
            description="Schd Trxn No Av Bal Pmt To Payroll",
            amount=0.0,
            user_id=user_id,
            file_id=uf.id,
        )
        junk_b = Transaction(
            date=datetime(2026, 3, 1),
            description="Schd Trxn No Av Bal Pmt To Festus",
            amount=0.0,
            user_id=user_id,
            file_id=uf.id,
        )
        real = Transaction(
            date=datetime(2026, 3, 2),
            description="Supplier payment",
            amount=-50.0,
            user_id=user_id,
            file_id=uf.id,
        )
        flagged_zero = Transaction(
            date=datetime(2026, 3, 3),
            description="Fee waiver pending",
            amount=0.0,
            user_id=user_id,
            file_id=uf.id,
        )
        db.session.add_all([junk_a, junk_b, real, flagged_zero])
        db.session.commit()

        assert find_junk_transaction_ids(file_id=uf.id) == [junk_a.id, junk_b.id]

        preview = cleanup_junk_transactions(dry_run=True, file_id=uf.id)
        assert preview.matched == 2
        assert preview.deleted == 0
        assert Transaction.query.filter_by(file_id=uf.id).count() == 4

        applied = cleanup_junk_transactions(dry_run=False, file_id=uf.id)
        assert applied.deleted == 2
        remaining = Transaction.query.filter_by(file_id=uf.id).order_by(Transaction.id).all()
        assert len(remaining) == 2
        assert {t.description for t in remaining} == {"Supplier payment", "Fee waiver pending"}
