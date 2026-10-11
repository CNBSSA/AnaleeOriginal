"""Production-equivalent schema path: ``flask db upgrade`` on PostgreSQL.

Exercised in CI (``.github/workflows/ci.yml`` ``migrate-postgres`` job). Locally,
set ``ANALEE_POSTGRES_MIGRATION_TEST=1`` and a ``postgresql://`` ``DATABASE_URL``
after running::

    ANALEE_SKIP_BOOT_SCHEMA=1 FLASK_APP=migrate_app.py flask db upgrade
"""
import os

import pytest
from sqlalchemy import create_engine, inspect

pytestmark = pytest.mark.skipif(
    os.environ.get('ANALEE_POSTGRES_MIGRATION_TEST') != '1',
    reason='PostgreSQL migration gate runs in CI only',
)

# Tables that must exist after ``flask db upgrade`` on an empty database.
_EXPECTED_TABLES = frozenset({
    'user',
    'entity',
    'account',
    'transaction',
    'uploaded_file',
    'company_settings',
    'admin_chart_of_accounts',
    'login_attempt',
    'trial_balance_approval',
    'practice_link',
    'practice_client_meta',
    'club_member_link',
    'alembic_version',
})


def _postgres_url():
    url = os.environ.get('DATABASE_URL', '')
    if not url.startswith('postgresql'):
        pytest.skip('DATABASE_URL is not PostgreSQL')
    return url


def test_fresh_database_has_expected_tables_after_upgrade():
    """CI runs ``flask db upgrade`` before pytest; tables must match models."""
    engine = create_engine(_postgres_url())
    tables = set(inspect(engine).get_table_names())
    missing = sorted(_EXPECTED_TABLES - tables)
    assert not missing, f'tables missing after migration upgrade: {missing}'


def test_alembic_version_is_at_head():
    from flask_migrate import current

    os.environ.setdefault('ANALEE_SKIP_BOOT_SCHEMA', '1')
    os.environ.setdefault('FLASK_APP', 'migrate_app.py')

    from migrate_app import app

    with app.app_context():
        script = current()
        assert script is not None
        head = script.get_current_head()
        assert head == 'd5e1f7a9c2b4', f'expected migration head d5e1f7a9c2b4, got {head}'
