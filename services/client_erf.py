"""ERF: find similar unexplained transactions on the same bank file."""
from __future__ import annotations

from difflib import SequenceMatcher

from services.client_explanation import unexplained_client_queue

SIMILARITY_THRESHOLD = 0.70
MAX_BATCH = 25


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or '').lower(), (b or '').lower()).ratio()


def find_similar_unexplained(
    file_id: int,
    user_id: int,
    description: str,
    *,
    exclude_id: int | None = None,
    threshold: float = SIMILARITY_THRESHOLD,
    limit: int = MAX_BATCH,
):
    description = (description or '').strip()
    if not description:
        return []
    matches = []
    for txn in unexplained_client_queue(file_id, user_id):
        if exclude_id is not None and txn.id == exclude_id:
            continue
        score = _ratio(description, txn.description or '')
        if score >= threshold:
            matches.append((score, txn))
    matches.sort(key=lambda item: item[0], reverse=True)
    return [txn for _, txn in matches[:limit]]


def serialize_similar(rows, *, reference_description: str) -> list[dict]:
    return [
        {
            'id': txn.id,
            'description': txn.description or '',
            'date': txn.date.strftime('%Y-%m-%d') if txn.date else '',
            'amount': f'{abs(txn.amount):,.2f}',
            'similarity': round(_ratio(reference_description, txn.description or ''), 3),
        }
        for txn in rows
    ]
