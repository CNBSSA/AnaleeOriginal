"""Boot-time schema guards for entity charts and company_settings.

Analee deploys use db.create_all() without flask db upgrade. Any column added to
models after the initial deploy must be healed here or CompanySettings queries
500 with "column ... does not exist" (see main hotfix e79e480 for logo).
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


def ensure_company_settings_schema() -> bool:
    """Heal company_settings columns (logo, entity_id) on existing databases."""
    try:
        insp = inspect(db.engine)
        if 'company_settings' not in insp.get_table_names():
            return True

        cols = {c['name'] for c in insp.get_columns('company_settings')}
        dialect = db.engine.dialect.name
        blob = 'BYTEA' if dialect == 'postgresql' else 'BLOB'
        to_add: list[tuple[str, str]] = []
        if 'logo' not in cols:
            to_add.append(('logo', blob))
        if 'logo_type' not in cols:
            to_add.append(('logo_type', 'VARCHAR(50)'))
        if 'entity_id' not in cols:
            to_add.append(('entity_id', 'INTEGER'))

        if to_add:
            with db.engine.begin() as conn:
                for col, coltype in to_add:
                    conn.execute(text(
                        f'ALTER TABLE company_settings ADD COLUMN {col} {coltype}'
                    ))
            logger.info(
                'Added missing company_settings columns: %s',
                [name for name, _ in to_add],
            )
        return True
    except Exception as exc:
        logger.error('company_settings schema guard failed: %s', exc)
        return False


def ensure_entity_chart_schema() -> bool:
    """Apply entity chart DDL on existing databases. Returns True if ready."""
    if not ensure_company_settings_schema():
        return False

    try:
        Entity.__table__.create(db.engine, checkfirst=True)
        _seed_entity_rows()

        insp = inspect(db.engine)
        tables = set(insp.get_table_names())

        if 'admin_chart_of_accounts' in tables:
            try:
                _ensure_admin_chart_entity_column(insp)
            except Exception as exc:
                db.session.rollback()
                logger.error('Admin chart entity column guard failed: %s', exc)

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
        with db.engine.begin() as conn:
            conn.execute(text(
                'ALTER TABLE admin_chart_of_accounts '
                'ADD COLUMN entity_id INTEGER'
            ))
        logger.info('Added admin_chart_of_accounts.entity_id column')
        insp = inspect(db.engine)

    private_id = db.session.execute(text(
        'SELECT id FROM entity WHERE name = :name'
    ), {'name': DEFAULT_ENTITY_NAME}).scalar()
    if private_id is None:
        return

    with db.engine.begin() as conn:
        conn.execute(text(
            'UPDATE admin_chart_of_accounts '
            'SET entity_id = :eid WHERE entity_id IS NULL'
        ), {'eid': private_id})

    try:
        _migrate_admin_chart_unique_constraint(inspect(db.engine))
    except Exception as exc:
        logger.error(
            'Admin chart unique constraint migration skipped: %s', exc
        )


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
        logger.warning(
            'admin_chart_of_accounts still has link-only unique on SQLite; '
            'run flask db upgrade if multi-entity seed fails'
        )
        return

    link_only = [
        uc['name'] for uc in uniques
        if uc['column_names'] == ['link'] and uc['name']
    ]
    with db.engine.begin() as conn:
        for name in link_only:
            conn.execute(text(
                f'ALTER TABLE admin_chart_of_accounts DROP CONSTRAINT "{name}"'
            ))
            logger.info(
                'Dropped legacy unique constraint %s on admin_chart_of_accounts.link',
                name,
            )
        conn.execute(text(
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


def prepare_subscriber_chart_access() -> bool:
    """Heal schema and seed charts before subscriber settings/import routes."""
    if not ensure_entity_chart_schema():
        return False
    try:
        from services.chart_of_accounts import seed_entities, seed_admin_charts
        seed_entities()
        seed_admin_charts()
    except Exception as exc:
        logger.error('Chart seed during subscriber route prep failed: %s', exc)
    return True
