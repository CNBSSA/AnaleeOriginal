"""Remove persisted statement-notice rows that predate the Unit 1.7 junk filter.

Only rows that the import filter would DROP today are removed — zero-amount
lines matching definite notice patterns in ``extraction_junk_filter``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from sqlalchemy.orm import Session

from models import Transaction
from services.extraction_junk_filter import JunkVerdict, classify_extraction_row

_ZERO_AMOUNT_BAND = 0.005


def transaction_is_definite_junk(description: str | None, amount) -> bool:
    """True when this persisted row matches a definite junk notice line."""
    return classify_extraction_row(description, amount) == JunkVerdict.DROP


@dataclass
class JunkCleanupReport:
    scanned_zero_amount: int = 0
    matched_junk: int = 0
    deleted: int = 0
    user_ids: set[int] = field(default_factory=set)
    sample_descriptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scanned_zero_amount": self.scanned_zero_amount,
            "matched_junk": self.matched_junk,
            "deleted": self.deleted,
            "user_ids": sorted(self.user_ids),
            "sample_descriptions": self.sample_descriptions,
        }


def _zero_amount_candidates(session: Session, user_ids: Sequence[int] | None):
    q = session.query(Transaction).filter(
        Transaction.amount > -_ZERO_AMOUNT_BAND,
        Transaction.amount < _ZERO_AMOUNT_BAND,
    )
    if user_ids:
        q = q.filter(Transaction.user_id.in_(list(user_ids)))
    return q


def find_definite_junk_transactions(
    session: Session,
    *,
    user_ids: Sequence[int] | None = None,
) -> list[Transaction]:
    """Return persisted transactions that are definite junk under Unit 1.7 rules."""
    junk: list[Transaction] = []
    for txn in _zero_amount_candidates(session, user_ids):
        if transaction_is_definite_junk(txn.description, txn.amount):
            junk.append(txn)
    return junk


def cleanup_definite_junk_transactions(
    session: Session,
    *,
    user_ids: Sequence[int] | None = None,
    apply: bool = False,
    sample_limit: int = 10,
) -> JunkCleanupReport:
    """Scan zero-amount rows and delete definite junk (dry-run unless ``apply``)."""
    report = JunkCleanupReport()
    to_delete: list[Transaction] = []

    for txn in _zero_amount_candidates(session, user_ids):
        report.scanned_zero_amount += 1
        if not transaction_is_definite_junk(txn.description, txn.amount):
            continue
        report.matched_junk += 1
        report.user_ids.add(txn.user_id)
        if len(report.sample_descriptions) < sample_limit:
            report.sample_descriptions.append(txn.description)
        to_delete.append(txn)

    if apply and to_delete:
        for txn in to_delete:
            session.delete(txn)
        session.commit()
        report.deleted = len(to_delete)
    else:
        session.rollback()

    return report
