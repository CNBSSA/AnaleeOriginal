"""Non-frozen helpers for the Analyze Data list page (#11033).

The frozen ``services.analyze_processing`` module is called only through its
public API. List-page performance (one bank-account lookup and one grouped
unprocessed-count query per request) lives here.
"""
from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy import func, or_

from models import Transaction, UploadedFile, db
from services import analyze_processing as ap


def _needs_processing_filter(bank_ids: List[int]):
    """SQL twin of ``analyze_processing._needs_processing_clause`` for list queries."""
    missing_account = Transaction.account_id.is_(None)
    if bank_ids:
        missing_account = or_(missing_account, Transaction.account_id.in_(bank_ids))
    return or_(
        missing_account,
        Transaction.explanation.is_(None),
        Transaction.explanation == '',
    )


def file_summaries_for_analyze_list(user_id: int) -> List[Dict[str, Any]]:
    """Uploaded files with transaction counts for the analyze list page."""
    bank_ids = ap.bank_account_ids_for_user(user_id)
    files = (
        UploadedFile.query.filter_by(user_id=user_id)
        .order_by(UploadedFile.upload_date.desc())
        .all()
    )

    counts = dict(
        db.session.query(Transaction.file_id, func.count(Transaction.id))
        .filter(Transaction.user_id == user_id)
        .group_by(Transaction.file_id)
        .all()
    )

    unprocessed_map = dict(
        db.session.query(Transaction.file_id, func.count(Transaction.id))
        .filter(
            Transaction.user_id == user_id,
            _needs_processing_filter(bank_ids),
        )
        .group_by(Transaction.file_id)
        .all()
    )

    summaries = []
    for uploaded_file in files:
        transaction_count = counts.get(uploaded_file.id, 0)
        unprocessed = unprocessed_map.get(uploaded_file.id, 0)
        summaries.append({
            'file': uploaded_file,
            'transaction_count': transaction_count,
            'unprocessed_count': unprocessed,
            'processed_count': max(0, transaction_count - unprocessed),
        })
    return summaries
