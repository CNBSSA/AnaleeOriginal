"""Helpers for paginated, phased transaction analysis."""
import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, or_

from models import Account, Transaction, UploadedFile, db

logger = logging.getLogger(__name__)

ANALYZE_PAGE_SIZE = 10
ANALYZE_BATCH_SIZE = 10


def transaction_needs_processing(transaction: Transaction) -> bool:
    """True when a row still needs account assignment or explanation."""
    return transaction.account_id is None and not (
        transaction.explanation and transaction.explanation.strip()
    )


def get_file_for_user(file_id: int, user_id: int) -> Optional[UploadedFile]:
    return UploadedFile.query.filter_by(id=file_id, user_id=user_id).first()


def count_file_transactions(file_id: int, user_id: int) -> int:
    return Transaction.query.filter_by(file_id=file_id, user_id=user_id).count()


def count_unprocessed_transactions(file_id: int, user_id: int) -> int:
    return Transaction.query.filter(
        Transaction.file_id == file_id,
        Transaction.user_id == user_id,
        Transaction.account_id.is_(None),
        or_(Transaction.explanation.is_(None), Transaction.explanation == ''),
    ).count()


def get_paginated_transactions(
    file_id: int,
    user_id: int,
    page: int,
    per_page: int = ANALYZE_PAGE_SIZE,
) -> Tuple[List[Transaction], int, int]:
    """Return (rows, total_count, total_pages) for the requested page."""
    page = max(1, page)
    base_query = Transaction.query.filter_by(
        file_id=file_id,
        user_id=user_id,
    ).order_by(Transaction.date, Transaction.id)

    total_count = base_query.count()
    total_pages = max(1, (total_count + per_page - 1) // per_page)
    page = min(page, total_pages)

    transactions = (
        base_query.offset((page - 1) * per_page).limit(per_page).all()
    )
    return transactions, total_count, total_pages


def save_analyze_form_transactions(user_id: int, form_data) -> int:
    """Persist account/explanation edits from the analyze form. Returns rows saved."""
    saved = 0
    transaction_ids = set()

    for key in form_data:
        if key.startswith('account_'):
            transaction_ids.add(int(key.split('_', 1)[1]))
        elif key.startswith('explanation_'):
            transaction_ids.add(int(key.split('_', 1)[1]))

    for transaction_id in transaction_ids:
        transaction = Transaction.query.filter_by(
            id=transaction_id,
            user_id=user_id,
        ).first()
        if not transaction:
            continue

        account_key = f'account_{transaction_id}'
        explanation_key = f'explanation_{transaction_id}'

        account_value = form_data.get(account_key, '').strip()
        if account_value:
            account = Account.query.filter_by(
                id=int(account_value),
                user_id=user_id,
            ).first()
            if account:
                transaction.account_id = account.id

        if explanation_key in form_data:
            transaction.explanation = form_data.get(explanation_key, '').strip()

        saved += 1

    if saved:
        db.session.commit()
    return saved


def process_transaction_batch(
    file_id: int,
    user_id: int,
    offset: int = 0,
    batch_size: int = ANALYZE_BATCH_SIZE,
    auto_apply_threshold: float = 0.85,
) -> Dict[str, Any]:
    """Process up to batch_size unprocessed transactions with account suggestions."""
    batch_size = max(1, min(batch_size, ANALYZE_BATCH_SIZE))
    offset = max(0, offset)

    unprocessed_query = Transaction.query.filter(
        Transaction.file_id == file_id,
        Transaction.user_id == user_id,
        Transaction.account_id.is_(None),
        or_(Transaction.explanation.is_(None), Transaction.explanation == ''),
    ).order_by(Transaction.date, Transaction.id)

    total_unprocessed = unprocessed_query.count()
    transactions = unprocessed_query.offset(offset).limit(batch_size).all()

    accounts = Account.query.filter_by(user_id=user_id, is_active=True).all()
    account_by_name = {account.name.lower(): account for account in accounts}

    from predictive_features import PredictiveFeatures
    predictor = PredictiveFeatures()

    results: List[Dict[str, Any]] = []
    for transaction in transactions:
        suggestion = predictor.suggest_account(
            transaction.description,
            transaction.explanation or '',
        )

        applied_account_id = None
        applied_account_name = None
        confidence = suggestion.get('confidence', 0) if suggestion else 0

        if suggestion.get('success') and suggestion.get('account'):
            account_name = suggestion['account']
            matched = account_by_name.get(str(account_name).lower())
            if matched and confidence >= auto_apply_threshold:
                transaction.account_id = matched.id
                applied_account_id = matched.id
                applied_account_name = matched.name

        results.append({
            'transaction_id': transaction.id,
            'description': transaction.description,
            'suggestion': suggestion,
            'applied_account_id': applied_account_id,
            'applied_account_name': applied_account_name,
        })

    if results:
        db.session.commit()

    processed_count = len(results)
    remaining = max(0, total_unprocessed - offset - processed_count)

    return {
        'success': True,
        'processed': processed_count,
        'offset': offset,
        'next_offset': offset + processed_count,
        'total_unprocessed': total_unprocessed,
        'remaining': remaining,
        'has_more': remaining > 0,
        'results': results,
    }


def file_summaries_for_user(user_id: int) -> List[Dict[str, Any]]:
    """Uploaded files with transaction counts for the analyze list page."""
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

    summaries = []
    for uploaded_file in files:
        transaction_count = counts.get(uploaded_file.id, 0)
        unprocessed = count_unprocessed_transactions(uploaded_file.id, user_id)
        summaries.append({
            'file': uploaded_file,
            'transaction_count': transaction_count,
            'unprocessed_count': unprocessed,
            'processed_count': max(0, transaction_count - unprocessed),
        })
    return summaries
