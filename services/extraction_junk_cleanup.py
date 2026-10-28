"""One-time cleanup for zero-amount extraction junk already persisted as transactions.

Unit 1.7 drops these rows at import time; statements imported earlier may still
have them. Only rows that match the same definite-junk rules as import are removed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from models import Transaction, db
from services.extraction_junk_filter import JunkVerdict, amount_is_zero, classify_extraction_row


@dataclass(frozen=True)
class JunkCleanupStats:
    scanned: int = 0
    matched: int = 0
    deleted: int = 0
    dry_run: bool = True

    def as_dict(self) -> dict:
        return {
            "scanned": self.scanned,
            "matched": self.matched,
            "deleted": self.deleted,
            "dry_run": self.dry_run,
        }


def iter_zero_amount_transactions(
    *,
    user_id: Optional[int] = None,
    file_id: Optional[int] = None,
) -> Iterable[Transaction]:
    """Yield transactions whose amount is effectively zero (candidate junk rows)."""
    query = Transaction.query
    if user_id is not None:
        query = query.filter(Transaction.user_id == user_id)
    if file_id is not None:
        query = query.filter(Transaction.file_id == file_id)
    for txn in query.order_by(Transaction.id):
        if amount_is_zero(txn.amount):
            yield txn


def is_definite_extraction_junk(description: str | None, amount) -> bool:
    return classify_extraction_row(description, amount) == JunkVerdict.DROP


def find_junk_transaction_ids(
    *,
    user_id: Optional[int] = None,
    file_id: Optional[int] = None,
) -> list[int]:
    ids: list[int] = []
    for txn in iter_zero_amount_transactions(user_id=user_id, file_id=file_id):
        if is_definite_extraction_junk(txn.description, txn.amount):
            ids.append(txn.id)
    return ids


def cleanup_junk_transactions(
    *,
    dry_run: bool = True,
    user_id: Optional[int] = None,
    file_id: Optional[int] = None,
    commit: bool = True,
) -> JunkCleanupStats:
    """Remove persisted rows that the import junk filter would drop today."""
    matched_ids: list[int] = []
    scanned = 0
    for txn in iter_zero_amount_transactions(user_id=user_id, file_id=file_id):
        scanned += 1
        if is_definite_extraction_junk(txn.description, txn.amount):
            matched_ids.append(txn.id)

    deleted = 0
    if not dry_run and matched_ids:
        deleted = (
            Transaction.query.filter(Transaction.id.in_(matched_ids)).delete(
                synchronize_session=False
            )
            or 0
        )
        if commit:
            db.session.commit()

    return JunkCleanupStats(
        scanned=scanned,
        matched=len(matched_ids),
        deleted=deleted,
        dry_run=dry_run,
    )


def sample_matched_descriptions(
    *,
    user_id: Optional[int] = None,
    file_id: Optional[int] = None,
    limit: int = 20,
) -> Sequence[str]:
    """Human-readable preview for dry-run output."""
    out: list[str] = []
    for txn in iter_zero_amount_transactions(user_id=user_id, file_id=file_id):
        if is_definite_extraction_junk(txn.description, txn.amount):
            out.append(txn.description)
            if len(out) >= limit:
                break
    return out
