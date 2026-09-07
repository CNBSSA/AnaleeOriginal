"""One accountant decision, applied to every matching row on the statement.

After automation runs, what is left is the exceptions — and those are handled
one row at a time. But exceptions arrive in *families*: forty card purchases at
the same supermarket, twelve identical debit orders. Deciding one and then
retyping it thirty-nine times is the largest remaining piece of manual work in
the product.

The client wizard has had this for a while (``services/client_erf.py``: explain
one, offer the similar ones). The accountant — who does the professional bulk
of the work — has had nothing, and the client version only carries the
explanation, never the account. This is the accountant's equivalent, carrying
both.

Matching uses the same payee key as history matching
(:func:`services.history_matching.normalize_description`), so "apply to
similar" groups rows exactly the way next month's automatic matching will. What
the accountant fans out today is what the system recognises on its own
thereafter.

Safety
------
* Scoped to one file and one user; ids are re-validated on apply, never trusted
  from the request.
* **A row a person already decided is never touched.** Any row whose
  explanation was written by a human — the accountant earlier, or the client —
  is excluded from the candidates entirely, so a fan-out cannot overwrite
  someone's considered judgement.
* Explanations are written through ``save_explanation``, which independently
  refuses to let an accountant overwrite a client's words.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from models import Account, Transaction, db
from services.client_explanation import (
    HUMAN_SOURCES,
    SOURCE_ACCOUNTANT,
    save_explanation,
)
from services.history_matching import normalize_description

logger = logging.getLogger(__name__)

#: Never fan out to more rows than a person can sanely eyeball at once.
MAX_FANOUT = 200


def find_matching_rows(
    file_id: int,
    user_id: int,
    transaction_id: int,
) -> Dict[str, Any]:
    """Rows on the same statement that share a payee with ``transaction_id``.

    Returns ``{'source': {...}, 'key': str, 'rows': [...]}``. ``rows`` is empty
    when the description carries nothing identifying (see
    ``normalize_description``) — better to offer nothing than to fan out on a
    key like "payment".
    """
    source = Transaction.query.filter_by(
        id=transaction_id, user_id=user_id, file_id=file_id).first()
    if source is None:
        return {'source': None, 'key': '', 'rows': []}

    key = normalize_description(source.description)
    if not key:
        return {'source': source, 'key': '', 'rows': []}

    candidates: List[Transaction] = []
    for txn in Transaction.query.filter(
        Transaction.file_id == file_id,
        Transaction.user_id == user_id,
        Transaction.id != source.id,
    ).order_by(Transaction.date, Transaction.id):
        if normalize_description(txn.description) != key:
            continue
        # A row a person already decided is off limits.
        if (getattr(txn, 'explanation_source', None) or '') in HUMAN_SOURCES:
            continue
        candidates.append(txn)
        if len(candidates) >= MAX_FANOUT:
            break

    return {'source': source, 'key': key, 'rows': candidates}


def apply_to_rows(
    file_id: int,
    user_id: int,
    transaction_ids: Sequence[int],
    account_id: Optional[int],
    explanation: str,
) -> Dict[str, Any]:
    """Apply one decision to the given rows. Returns what actually changed.

    ``transaction_ids`` are re-validated against the file, the user, and the
    "not decided by a person" rule — the request is never trusted.
    """
    result = {'accounts_set': 0, 'explanations_set': 0, 'skipped': 0}

    wanted = {int(i) for i in transaction_ids if str(i).lstrip('-').isdigit()}
    if not wanted:
        return result

    account = None
    if account_id:
        account = Account.query.filter_by(id=int(account_id), user_id=user_id).first()
        if account is None:
            # An account that is not this user's is not an error to guess
            # around — apply the explanation only.
            logger.warning("Fan-out ignored account %s: not this user's", account_id)

    text = (explanation or '').strip()

    rows = Transaction.query.filter(
        Transaction.id.in_(wanted),
        Transaction.file_id == file_id,
        Transaction.user_id == user_id,
    ).all()

    for txn in rows:
        if (getattr(txn, 'explanation_source', None) or '') in HUMAN_SOURCES:
            result['skipped'] += 1
            continue
        if account is not None:
            txn.account_id = account.id
            result['accounts_set'] += 1
        if text:
            saved, _reason = save_explanation(txn, text, SOURCE_ACCOUNTANT)
            if saved:
                result['explanations_set'] += 1

    result['skipped'] += len(wanted) - len(rows)
    db.session.commit()
    logger.info(
        "Fan-out on file %s: %s account(s), %s explanation(s), %s skipped",
        file_id, result['accounts_set'], result['explanations_set'], result['skipped'])
    return result


def serialize_rows(rows: Sequence[Transaction]) -> List[Dict[str, Any]]:
    return [{
        'id': txn.id,
        'date': txn.date.strftime('%Y-%m-%d') if txn.date else '',
        'description': txn.description or '',
        'amount': float(txn.amount) if txn.amount is not None else 0.0,
        'has_account': txn.account_id is not None,
        'has_explanation': bool((txn.explanation or '').strip()),
    } for txn in rows]
