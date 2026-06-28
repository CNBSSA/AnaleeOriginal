"""Tests for boot-time entity chart schema guard."""
from sqlalchemy import inspect, text

from models import db, Entity, AdminChartOfAccounts
from services.entity_chart_schema import (
    ensure_entity_chart_schema,
    default_entity_id,
    DEFAULT_ENTITY_NAME,
)


def test_ensure_schema_on_fresh_db(app):
    with app.app_context():
        assert ensure_entity_chart_schema() is True
        assert Entity.query.count() == 5
        assert default_entity_id() is not None
        private = Entity.query.filter_by(name=DEFAULT_ENTITY_NAME).first()
        assert default_entity_id() == private.id


def test_schema_guard_adds_entity_id_to_legacy_admin_table(app):
    """Simulate pre-migration DB: admin chart exists without entity_id."""
    with app.app_context():
        db.drop_all()
        db.session.execute(text(
            'CREATE TABLE entity (id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL UNIQUE)'
        ))
        db.session.execute(text(
            "INSERT INTO entity (name) VALUES ('Private Company')"
        ))
        db.session.execute(text('''
            CREATE TABLE admin_chart_of_accounts (
                id INTEGER PRIMARY KEY,
                link VARCHAR(20) NOT NULL UNIQUE,
                code VARCHAR(20) NOT NULL,
                name VARCHAR(100) NOT NULL,
                category VARCHAR(50) NOT NULL,
                sub_category VARCHAR(50),
                description TEXT
            )
        '''))
        db.session.execute(text('''
            INSERT INTO admin_chart_of_accounts
            (link, code, name, category, sub_category)
            VALUES ('ca.810.001', '1010', 'Bank Cheque Account 1', 'Assets', 'Current Asset')
        '''))
        db.session.commit()

        assert ensure_entity_chart_schema() is True

        insp = inspect(db.engine)
        cols = {c['name'] for c in insp.get_columns('admin_chart_of_accounts')}
        assert 'entity_id' in cols

        row = db.session.execute(text(
            'SELECT entity_id FROM admin_chart_of_accounts WHERE link = :link'
        ), {'link': 'ca.810.001'}).scalar()
        private_id = Entity.query.filter_by(name=DEFAULT_ENTITY_NAME).first().id
        assert row == private_id
