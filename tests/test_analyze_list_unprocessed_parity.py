"""Work #11035 / audit — analyze list unprocessed counts match frozen engine API."""
from __future__ import annotations

from datetime import datetime

import pytest

from models import Transaction, UploadedFile, db
from services.analyze_list_page import file_summaries_for_analyze_list
from services.analyze_processing import count_unprocessed_transactions
from test_analyze_processing import (
    _add_account,
    _add_bank_account,
    _add_transactions,
    analyze_file,
    analyze_user,
)


def _assert_parity_for_user(app, user_id: int) -> None:
    with app.app_context():
        summaries = file_summaries_for_analyze_list(user_id)
        files = UploadedFile.query.filter_by(user_id=user_id).order_by(UploadedFile.id).all()
        assert len(summaries) == len(files)
        by_id = {s['file'].id: s for s in summaries}
        for uploaded in files:
            summary = by_id[uploaded.id]
            engine_count = count_unprocessed_transactions(uploaded.id, user_id)
            assert summary['unprocessed_count'] == engine_count, (
                f'file {uploaded.id}: list={summary["unprocessed_count"]} '
                f'engine={engine_count}'
            )
            assert (
                summary['processed_count'] + summary['unprocessed_count']
                == summary['transaction_count']
            )


def test_analyze_list_unprocessed_parity_single_file(app, analyze_user, analyze_file):
    _add_transactions(app, analyze_user, analyze_file, count=3)
    account_id = _add_account(app, analyze_user)
    with app.app_context():
        first = Transaction.query.filter_by(file_id=analyze_file).first()
        first.account_id = account_id
        db.session.commit()
    _assert_parity_for_user(app, analyze_user)


def test_analyze_list_unprocessed_parity_import_like_bank_stamp(
    app, analyze_user, analyze_file,
):
    """Import paths stamp ca.810.* — rows must stay unprocessed until explained."""
    _add_transactions(app, analyze_user, analyze_file, count=5)
    bank_account_id = _add_bank_account(app, analyze_user)
    with app.app_context():
        for txn in Transaction.query.filter_by(file_id=analyze_file).all():
            txn.account_id = bank_account_id
        db.session.commit()
    _assert_parity_for_user(app, analyze_user)


def test_analyze_list_unprocessed_parity_multi_file(app, analyze_user):
    with app.app_context():
        file_a = UploadedFile(filename='a.xlsx', user_id=analyze_user)
        file_b = UploadedFile(filename='b.xlsx', user_id=analyze_user)
        db.session.add_all([file_a, file_b])
        db.session.commit()
        fid_a, fid_b = file_a.id, file_b.id

    _add_transactions(app, analyze_user, fid_a, count=4)
    _add_transactions(app, analyze_user, fid_b, count=2)
    expense_id = _add_account(app, analyze_user)
    bank_id = _add_bank_account(app, analyze_user)

    with app.app_context():
        for txn in Transaction.query.filter_by(file_id=fid_a).limit(2):
            txn.account_id = expense_id
            txn.explanation = 'Done'
        for txn in Transaction.query.filter_by(file_id=fid_b).all():
            txn.account_id = bank_id
        db.session.commit()

    _assert_parity_for_user(app, analyze_user)
