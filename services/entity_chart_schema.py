"""Boot-time schema guard for entity-scoped charts.

Analee deploys use db.create_all() and do not run flask db upgrade, so new
columns/tables from the entity chart feature must be applied idempotently at
startup (same pattern as alert_history.alert_config_id in app.py).
"""
from __future__ import annotations

import logging

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from models import db, Entity

logger = logging.getLogger(__name__)

ENTITY_NAMES = (
    'Sole Proprietor',
    'Close Corporation',
    'Private Company',
    'NPO',
    'Partnership',
)
DEFAULT_ENTITY_NAME = 'Private Company'


def ensure_entity_chart_schema() -> bool:
    """Apply entity chart DDL on existing databases. Returns True if ready."""
    try:
        Entity.__table__.create(db.engine, checkfirst=True)
        _seed_entity_rows()

        insp = inspect(db.engine)
        tables = set(insp.get_table_names())

        if 'admin_chart_of_accounts' in tables:
            _ensure_admin_chart_entity_column(insp)

        if 'company_settings' in tables:
            _ensure_company_settings_entity_column(insp)

        db.session.commit()
        return True
    except SQLAlchemyError as exc:
        db.session.rollback()
        logger.error('Entity chart schema guard failed: %s', exc)
        return False
    except Exception as exc:
        db.session.rollback()
        logger.error('Entity chart schema guard failed: %s', exc)
        return False


def _seed_entity_rows() -> None:
    existing = {
        row[0]
        for row in db.session.execute(text('SELECT name FROM entity')).fetchall()
    }
    added = False
    for name in ENTITY_NAMES:
        if name in existing:
            continue
        db.session.execute(
            text('INSERT INTO entity (name) VALUES (:name)'),
            {'name': name},
        )
        logger.info('Seeded entity row: %s', name)
        added = True
    if added:
        db.session.commit()


def _ensure_admin_chart_entity_column(insp) -> None:
    cols = {c['name'] for c in insp.get_columns('admin_chart_of_accounts')}
    if 'entity_id' not in cols:
        db.session.execute(text(
            'ALTER TABLE admin_chart_of_accounts '
            'ADD COLUMN entity_id INTEGER'
        ))
        db.session.commit()
        logger.info('Added admin_chart_of_accounts.entity_id column')
        insp = inspect(db.engine)

    private_id = db.session.execute(text(
        'SELECT id FROM entity WHERE name = :name'
    ), {'name': DEFAULT_ENTITY_NAME}).scalar()
    if private_id is None:
        return

    db.session.execute(text(
        'UPDATE admin_chart_of_accounts '
        'SET entity_id = :eid WHERE entity_id IS NULL'
    ), {'eid': private_id})
    db.session.commit()

    _migrate_admin_chart_unique_constraint(insp)


def _ensure_company_settings_entity_column(insp) -> None:
    cols = {c['name'] for c in insp.get_columns('company_settings')}
    if 'entity_id' in cols:
        return
    db.session.execute(text(
        'ALTER TABLE company_settings ADD COLUMN entity_id INTEGER'
    ))
    logger.info('Added company_settings.entity_id column')


def _migrate_admin_chart_unique_constraint(insp) -> None:
    """Replace global link unique with (entity_id, link) when still on old shape."""
    uniques = insp.get_unique_constraints('admin_chart_of_accounts')
    has_composite = any(
        set(uc['column_names']) == {'entity_id', 'link'}
        for uc in uniques
    )
    if has_composite:
        return

    dialect = db.engine.dialect.name
    if dialect == 'sqlite':
        # SQLite rebuild is handled by Alembic; skip at runtime.
        logger.warning(
            'admin_chart_of_accounts still has link-only unique on SQLite; '
            'run flask db upgrade if multi-entity seed fails'
        )
        return

    link_only = [
        uc['name'] for uc in uniques
        if uc['column_names'] == ['link'] and uc['name']
    ]
    for name in link_only:
        db.session.execute(text(
            f'ALTER TABLE admin_chart_of_accounts DROP CONSTRAINT "{name}"'
        ))
        logger.info(
            'Dropped legacy unique constraint %s on admin_chart_of_accounts.link',
            name,
        )

    db.session.execute(text(
        'ALTER TABLE admin_chart_of_accounts '
        'ADD CONSTRAINT uq_admin_coa_entity_link UNIQUE (entity_id, link)'
    ))
    logger.info('Created uq_admin_coa_entity_link on admin_chart_of_accounts')


def default_entity_id() -> int | None:
    """Primary key of Private Company, or first entity if missing."""
    row = db.session.execute(text(
        'SELECT id FROM entity WHERE name = :name'
    ), {'name': DEFAULT_ENTITY_NAME}).first()
    if row:
        return row[0]
    row = db.session.execute(text('SELECT id FROM entity ORDER BY id LIMIT 1')).first()
    return row[0] if row else None
