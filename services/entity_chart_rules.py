"""Entity type change rules — BooksXperts parity for AnaleeOriginal."""
from __future__ import annotations

from models import Account, CompanySettings, Transaction, db
from services.chart_of_accounts import set_entity_for_user

ENTITY_CHANGE_LOCKED_MESSAGE = (
    'Entity type cannot be changed because transactions have been posted. '
    'Delete all transactions and uploaded data first, then change the entity type '
    'to refresh your chart of accounts.'
)


class EntityChangeBlocked(Exception):
    """Raised when entity type cannot be changed for this user."""


def user_has_posted_transactions(user_id: int) -> bool:
    return Transaction.query.filter_by(user_id=user_id).count() > 0


def can_change_entity(user_id: int, new_entity_id: int) -> tuple[bool, str]:
    settings = CompanySettings.query.filter_by(user_id=user_id).first()
    current_entity_id = settings.entity_id if settings else None

    if not new_entity_id:
        return False, 'Please select an entity type.'

    if current_entity_id == new_entity_id:
        return True, ''

    if user_has_posted_transactions(user_id):
        return False, ENTITY_CHANGE_LOCKED_MESSAGE

    return True, ''


def reset_user_chart(user_id: int) -> int:
    """Remove a user's chart accounts so it can be rebuilt for a new entity."""
    deleted = Account.query.filter_by(user_id=user_id).delete()
    db.session.flush()
    return deleted


def apply_entity_change(user_id: int, entity_id: int) -> tuple[int, bool]:
    """Apply entity selection and (re)build the user chart.

    Returns (accounts_added, chart_rebuilt).
    Raises EntityChangeBlocked when transactions prevent a change.
    """
    allowed, message = can_change_entity(user_id, entity_id)
    if not allowed:
        raise EntityChangeBlocked(message)

    settings = CompanySettings.query.filter_by(user_id=user_id).first()
    current_entity_id = settings.entity_id if settings else None
    entity_changed = current_entity_id != entity_id

    if entity_changed:
        reset_user_chart(user_id)

    added = set_entity_for_user(user_id, entity_id)
    return added, entity_changed


def provision_chart_if_missing(user_id: int) -> int:
    """Backfill chart for users who have entity set but no accounts yet."""
    settings = CompanySettings.query.filter_by(user_id=user_id).first()
    if not settings or not settings.entity_id:
        return 0

    if Account.query.filter_by(user_id=user_id).count() > 0:
        return Account.query.filter_by(user_id=user_id).count()

    return set_entity_for_user(user_id, settings.entity_id)
