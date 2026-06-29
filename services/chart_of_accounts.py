"""Entity-scoped chart of accounts — master seed + per-user copy (BooksXperts parity)."""
from __future__ import annotations

import logging

from models import db, Entity, AdminChartOfAccounts, Account, CompanySettings
from services.chart_seed_data import ENTITY_NAMES, COMMON_ACCOUNTS, ENTITY_SPECIFIC

logger = logging.getLogger(__name__)

DEFAULT_ENTITY_NAME = 'Private Company'


def seed_entities() -> dict[str, Entity]:
    """Ensure the five SA entity types exist. Returns name → Entity."""
    entities: dict[str, Entity] = {}
    for name in ENTITY_NAMES:
        ent = Entity.query.filter_by(name=name).first()
        if not ent:
            ent = Entity(name=name)
            db.session.add(ent)
            db.session.flush()
        entities[name] = ent
    db.session.commit()
    return entities


def seed_admin_charts() -> tuple[int, int]:
    """Seed AdminChartOfAccounts from BooksXperts chart data. Idempotent."""
    entities = seed_entities()
    created = skipped = 0

    for entity_name, extras in ENTITY_SPECIFIC.items():
        entity = entities[entity_name]
        rows = COMMON_ACCOUNTS + extras
        for num, name, link, category, sub_category in rows:
            code = str(num)
            existing = AdminChartOfAccounts.query.filter_by(
                entity_id=entity.id, link=link
            ).first()
            if existing:
                skipped += 1
                continue
            db.session.add(AdminChartOfAccounts(
                entity_id=entity.id,
                link=link,
                code=code,
                name=name,
                category=category,
                sub_category=sub_category,
            ))
            created += 1

    db.session.commit()
    logger.info('Admin chart seed: %s created, %s skipped', created, skipped)
    return created, skipped


def set_entity_for_user(user_id: int, entity_id: int) -> int:
    """Copy the master chart for entity_id into the user's Account table.

    Idempotent — skips links the user already has. Returns count of new accounts.
    """
    entity = Entity.query.get(entity_id)
    if not entity:
        raise ValueError(f'Unknown entity id {entity_id}')

    settings = CompanySettings.query.filter_by(user_id=user_id).first()
    if not settings:
        settings = CompanySettings(
            user_id=user_id,
            company_name='My Company',
            financial_year_end=2,
        )
        db.session.add(settings)

    settings.entity_id = entity_id

    existing_links = {
        a.link
        for a in Account.query.filter_by(user_id=user_id).all()
    }

    master_rows = AdminChartOfAccounts.query.filter_by(entity_id=entity_id).all()
    if not master_rows:
        seed_admin_charts()
        master_rows = AdminChartOfAccounts.query.filter_by(entity_id=entity_id).all()

    added = 0
    for row in master_rows:
        if row.link in existing_links:
            continue
        db.session.add(Account(
            link=row.link,
            name=row.name,
            category=row.category,
            sub_category=row.sub_category,
            account_code=row.code,
            user_id=user_id,
            is_active=True,
        ))
        existing_links.add(row.link)
        added += 1

    db.session.commit()
    return added


def provision_user_chart(user_id: int) -> int:
    """Default chart provisioning for subscriber approval.

    Uses the user's chosen entity from settings, else Private Company.
    """
    settings = CompanySettings.query.filter_by(user_id=user_id).first()
    if settings and settings.entity_id:
        return set_entity_for_user(user_id, settings.entity_id)

    entities = seed_entities()
    default_entity = entities[DEFAULT_ENTITY_NAME]
    return set_entity_for_user(user_id, default_entity.id)
