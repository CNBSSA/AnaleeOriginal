"""Helpers for paginated, phased transaction analysis."""
import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, or_

from models import Account, Transaction, UploadedFile, db

logger = logging.getLogger(__name__)

ANALYZE_PAGE_SIZE = 10
#: Rows per auto-process request. Raised from 10 now that a batch is ONE AI
#: call rather than one call per row — see services/bulk_suggestions.py.
ANALYZE_BATCH_SIZE = 25


def transaction_needs_processing(transaction: Transaction) -> bool:
    """True when a row still needs an account assignment OR an explanation.

    Previously this was an AND: a row counted as finished the moment it had
    *either* one. That interacted badly with import — both import paths stamp
    the chosen bank account onto every row (`bank_statements/services.py`, and
    the "assign all rows to account" selector on the OCR review screen), so a
    freshly imported statement was marked "All processed" before anything had
    been categorised or explained, and its rows were invisible to the
    exceptions view. A row is only done when both halves are present.
    """
    has_account = transaction.account_id is not None
    has_explanation = bool(transaction.explanation and transaction.explanation.strip())
    return not (has_account and has_explanation)


def get_file_for_user(file_id: int, user_id: int) -> Optional[UploadedFile]:
    return UploadedFile.query.filter_by(id=file_id, user_id=user_id).first()


def count_file_transactions(file_id: int, user_id: int) -> int:
    return Transaction.query.filter_by(file_id=file_id, user_id=user_id).count()


def _needs_processing_clause():
    """SQL twin of :func:`transaction_needs_processing` — missing EITHER half."""
    return or_(
        Transaction.account_id.is_(None),
        Transaction.explanation.is_(None),
        Transaction.explanation == '',
    )


def count_unprocessed_transactions(file_id: int, user_id: int) -> int:
    return Transaction.query.filter(
        Transaction.file_id == file_id,
        Transaction.user_id == user_id,
        _needs_processing_clause(),
    ).count()


def get_paginated_transactions(
    file_id: int,
    user_id: int,
    page: int,
    per_page: int = ANALYZE_PAGE_SIZE,
    only_unprocessed: bool = False,
) -> Tuple[List[Transaction], int, int]:
    """Return (rows, total_count, total_pages) for the requested page.

    ``only_unprocessed`` (Festus 2026-08-27, "review only the exceptions"):
    when True, restrict to rows that still need an account or explanation —
    the same predicate as ``transaction_needs_processing`` /
    ``count_unprocessed_transactions`` — so after a whole-statement pass the
    accountant sees the handful that need their eye, not all 1,469. Default
    False → byte-identical to before (full paginated list)."""
    page = max(1, page)
    base_query = Transaction.query.filter_by(
        file_id=file_id,
        user_id=user_id,
    )
    if only_unprocessed:
        base_query = base_query.filter(_needs_processing_clause())
    base_query = base_query.order_by(Transaction.date, Transaction.id)

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
            from services.client_explanation import CLIENT_SOURCES, SOURCE_ACCOUNTANT, save_explanation
            new_text = form_data.get(explanation_key, '').strip()
            if new_text:
                current_source = getattr(transaction, 'explanation_source', None) or ''
                # Keep what the client said: an accountant explanation must not
                # overwrite a client-provided one. The account assignment above is
                # the accountant's legitimate override and MUST still be saved, so
                # only skip the explanation write here — never `continue`, which
                # would also skip `saved += 1` and (when this is the only edited
                # row) the commit below, silently dropping the account change.
                if current_source not in CLIENT_SOURCES:
                    save_explanation(transaction, new_text, SOURCE_ACCOUNTANT)

        saved += 1

    # Commit unconditionally: an account change on a client-locked row is a real
    # edit even though it does not increment a "saved explanation" count, so the
    # commit must not be gated on `saved`. An empty commit is a harmless no-op.
    db.session.commit()
    return saved


def process_transaction_batch(
    file_id: int,
    user_id: int,
    offset: int = 0,
    batch_size: int = ANALYZE_BATCH_SIZE,
    auto_apply_threshold: float = 0.85,
) -> Dict[str, Any]:
    """Fill in both halves of up to batch_size rows in ONE AI call.

    Previously this made one Claude call per transaction, each carrying the
    user's entire chart of accounts (~1 000 accounts) — a 160-row statement
    therefore sent ~160 copies of that chart — and it only ever wrote the
    account, leaving every row it "processed" without an explanation. It now
    asks for both in a single batched call.

    Two rules that must not be relaxed:

    * An account is written ONLY where there is none. Both import paths stamp
      the chosen bank account onto every row, and that assignment must never be
      silently replaced by a suggestion.
    * An explanation is written ONLY where there is none, and always tagged
      ``SOURCE_AI`` so the books show a machine wrote it. ``save_explanation``
      refuses to let it overwrite anything a person wrote.
    """
    from services.bulk_suggestions import BULK_SUGGESTION_BATCH, suggest_for_rows
    from services.client_explanation import SOURCE_AI, save_explanation

    batch_size = max(1, min(batch_size, BULK_SUGGESTION_BATCH))
    offset = max(0, offset)

    # A row still needs work if it is missing EITHER half.
    unprocessed_query = Transaction.query.filter(
        Transaction.file_id == file_id,
        Transaction.user_id == user_id,
        _needs_processing_clause(),
    ).order_by(Transaction.date, Transaction.id)

    total_unprocessed = unprocessed_query.count()
    transactions = unprocessed_query.offset(offset).limit(batch_size).all()
    if not transactions:
        return {
            'success': True, 'processed': 0, 'offset': offset,
            'next_offset': offset, 'applied': 0, 'explained': 0,
            'total_unprocessed': total_unprocessed, 'remaining': 0,
            'has_more': False, 'results': [],
        }

    accounts = Account.query.filter_by(user_id=user_id, is_active=True).all()

    suggestions = suggest_for_rows(
        [{
            'index': index,
            'date': txn.date.strftime('%Y-%m-%d') if txn.date else '',
            'description': txn.description or '',
            'amount': txn.amount,
        } for index, txn in enumerate(transactions)],
        accounts,
    )

    results: List[Dict[str, Any]] = []
    for index, transaction in enumerate(transactions):
        suggestion = suggestions.get(index)

        applied_account_id = None
        applied_account_name = None
        explained = False

        if suggestion is not None:
            # Account: only into an empty slot, only above the gate.
            if (transaction.account_id is None
                    and suggestion.account_id is not None
                    and suggestion.confidence >= auto_apply_threshold):
                transaction.account_id = suggestion.account_id
                applied_account_id = suggestion.account_id
                applied_account_name = suggestion.account_name

            # Explanation: only into an empty slot, always attributed.
            if suggestion.explanation and not (transaction.explanation or '').strip():
                saved, _reason = save_explanation(
                    transaction, suggestion.explanation, SOURCE_AI)
                explained = saved

        results.append({
            'transaction_id': transaction.id,
            'description': transaction.description,
            'suggestion': {
                'success': suggestion is not None,
                'account': suggestion.account_name if suggestion else None,
                'confidence': suggestion.confidence if suggestion else 0,
                'explanation': suggestion.explanation if suggestion else '',
            },
            'applied_account_id': applied_account_id,
            'applied_account_name': applied_account_name,
            'explained': explained,
        })

    db.session.commit()

    processed_count = len(results)
    applied_count = sum(1 for r in results if r['applied_account_id'] is not None)
    explained_count = sum(1 for r in results if r['explained'])

    # A row leaves the result set only when BOTH halves are now present, so the
    # window must advance past exactly those rows that still need work.
    completed = sum(
        1 for index, txn in enumerate(transactions)
        if txn.account_id is not None and (txn.explanation or '').strip()
    )
    next_offset = offset + (processed_count - completed)
    remaining = max(0, total_unprocessed - completed - next_offset)

    return {
        'success': True,
        'processed': processed_count,
        'offset': offset,
        'next_offset': next_offset,
        'applied': applied_count,
        'explained': explained_count,
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
