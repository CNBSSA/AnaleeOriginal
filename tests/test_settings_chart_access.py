"""Tests for subscriber /settings chart access after legacy deploy."""
from sqlalchemy import inspect, text

from models import db, Account, CompanySettings, User
from services.entity_chart_schema import prepare_subscriber_chart_access


def _legacy_subscriber_db(app):
    """Prod-like DB: company_settings without logo/entity_id, account table only."""
    with app.app_context():
        db.drop_all()
        db.session.execute(text('''
            CREATE TABLE "user" (
                id INTEGER PRIMARY KEY,
                username VARCHAR(64) NOT NULL,
                email VARCHAR(120) NOT NULL,
                password_hash VARCHAR(256),
                is_admin BOOLEAN DEFAULT 0,
                is_deleted BOOLEAN DEFAULT 0,
                subscription_status VARCHAR(20) DEFAULT 'active'
            )
        '''))
        db.session.execute(text('''
            CREATE TABLE company_settings (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                company_name VARCHAR(200) NOT NULL,
                registration_number VARCHAR(50),
                tax_number VARCHAR(50),
                vat_number VARCHAR(50),
                address TEXT,
                financial_year_end INTEGER NOT NULL,
                created_at DATETIME,
                updated_at DATETIME
            )
        '''))
        db.session.execute(text('''
            CREATE TABLE account (
                id INTEGER PRIMARY KEY,
                link VARCHAR(20) NOT NULL,
                category VARCHAR(100) NOT NULL,
                sub_category VARCHAR(100),
                account_code VARCHAR(20),
                name VARCHAR(100) NOT NULL,
                user_id INTEGER NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                created_at DATETIME,
                updated_at DATETIME
            )
        '''))
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
            INSERT INTO "user" (id, username, email) VALUES (1, 'sub', 'sub@example.com')
        '''))
        db.session.execute(text('''
            INSERT INTO company_settings (id, user_id, company_name, financial_year_end)
            VALUES (1, 1, 'Acme Ltd', 2)
        '''))
        db.session.commit()


def test_prepare_subscriber_chart_access_heals_legacy_db(app):
    _legacy_subscriber_db(app)
    with app.app_context():
        assert prepare_subscriber_chart_access() is True
        insp = inspect(db.engine)
        cs_cols = {c['name'] for c in insp.get_columns('company_settings')}
        admin_cols = {c['name'] for c in insp.get_columns('admin_chart_of_accounts')}
        assert 'entity_id' in cs_cols
        assert 'entity_id' in admin_cols


def test_settings_queries_succeed_after_prepare(app):
    _legacy_subscriber_db(app)
    with app.app_context():
        assert prepare_subscriber_chart_access() is True
        company = CompanySettings.query.filter_by(user_id=1).first()
        assert company is not None
        accounts = Account.query.filter_by(user_id=1, is_active=True).all()
        assert isinstance(accounts, list)
