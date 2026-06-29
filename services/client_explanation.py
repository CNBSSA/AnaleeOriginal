"""Client explanation queue and precedence for standalone Analee."""
from __future__ import annotations

from sqlalchemy import or_

from models import Transaction, UploadedFile, db

SOURCE_ACCOUNTANT = 'accountant'
SOURCE_CLIENT = 'client'
SOURCE_CLIENT_ERF = 'client_erf'
CLIENT_SOURCES = frozenset({SOURCE_CLIENT, SOURCE_CLIENT_ERF})


def get_file_for_owner(file_id: int, user_id: int) -> UploadedFile | None:
    return UploadedFile.query.filter_by(id=file_id, user_id=user_id).first()


def unexplained_client_queue(file_id: int, user_id: int):
    return (
        Transaction.query.filter(
            Transaction.file_id == file_id,
            Transaction.user_id == user_id,
            or_(Transaction.explanation.is_(None), Transaction.explanation == ''),
        )
        .order_by(Transaction.date, Transaction.id)
    )


def queue_counts(file_id: int, user_id: int) -> tuple[int, int]:
    total = Transaction.query.filter_by(file_id=file_id, user_id=user_id).count()
    remaining = unexplained_client_queue(file_id, user_id).count()
    return remaining, total


def save_explanation(transaction: Transaction, text: str, source: str) -> tuple[bool, str]:
    cleaned = (text or '').strip()[:500]
    if not cleaned:
        return False, 'empty'
    current = getattr(transaction, 'explanation_source', None) or ''
    if source == SOURCE_ACCOUNTANT and current in CLIENT_SOURCES:
        return False, 'client_locked'
    transaction.explanation = cleaned
    transaction.explanation_source = source
    return True, 'ok'


def transaction_kind(transaction: Transaction) -> tuple[str, str]:
    if transaction.amount >= 0:
        return 'Deposit (money in)', 'deposit'
    return 'Payment (money out)', 'payment'


def serialize_transaction(transaction: Transaction) -> dict:
    kind, kind_class = transaction_kind(transaction)
    return {
        'id': transaction.id,
        'date': transaction.date.strftime('%Y-%m-%d') if transaction.date else '',
        'description': transaction.description or '',
        'amount': f'{abs(transaction.amount):,.2f}',
        'kind': kind,
        'kind_class': kind_class,
    }
