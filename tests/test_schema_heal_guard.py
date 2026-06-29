"""Tests for the boot-time schema-heal guard (app._heal_missing_columns).

The deploy builds schema with create_all() (no migrations), so a column added to
a model after its table already existed is missing in production. This guard adds
those columns on boot. These tests exercise it against a legacy SQLite schema.
"""
import sys
import types

# app.py imports these optional extensions at module load; stub them so the
# module (and the helper under test) imports on any environment.
for _name in ("flask_migrate", "flask_apscheduler"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
sys.modules["flask_migrate"].Migrate = lambda *a, **k: types.SimpleNamespace(
    init_app=lambda *a, **k: None)


class _Sched:
    def __init__(self, *a, **k):
        pass

    def init_app(self, *a, **k):
        pass


sys.modules["flask_apscheduler"].APScheduler = _Sched

from sqlalchemy import create_engine, inspect, text  # noqa: E402

from app import _heal_missing_columns  # noqa: E402
from models import User, Account, Transaction  # noqa: E402


def test_heal_adds_missing_columns_and_preserves_rows(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path/'legacy.db'}")
    with eng.begin() as c:
        # Legacy 'account' missing is_active / account_code / sub_category / updated_at.
        c.execute(text(
            'CREATE TABLE account (id INTEGER PRIMARY KEY, link VARCHAR(20), '
            'category VARCHAR(100), name VARCHAR(100), user_id INTEGER, '
            'created_at DATETIME)'))
        # Legacy 'user' (reserved word -> quoted) missing subscription_status etc.
        c.execute(text(
            'CREATE TABLE "user" (id INTEGER PRIMARY KEY, username VARCHAR(64), '
            'email VARCHAR(120), password_hash VARCHAR(256), created_at DATETIME)'))
        c.execute(text("INSERT INTO account (id, link, category, name, user_id) "
                       "VALUES (1, 'ca.810.001', 'Assets', 'Bank', 1)"))
        c.execute(text('INSERT INTO "user" (id, username, email) '
                       "VALUES (1, 'u', 'u@example.com')"))

    assert 'is_active' not in {c['name'] for c in inspect(eng).get_columns('account')}

    added = _heal_missing_columns(eng, [User, Account])

    acct_cols = {c['name'] for c in inspect(eng).get_columns('account')}
    user_cols = {c['name'] for c in inspect(eng).get_columns('user')}
    # Every model column now exists on the live tables.
    assert {col.name for col in Account.__table__.columns} <= acct_cols
    assert {col.name for col in User.__table__.columns} <= user_cols
    assert ('account', 'is_active') in added
    assert ('user', 'subscription_status') in added

    # Existing row preserved; the newly added column is NULL (not NOT NULL).
    with eng.begin() as c:
        assert c.execute(text('SELECT is_active FROM account WHERE id=1')).fetchone()[0] is None
        assert c.execute(text('SELECT name FROM account WHERE id=1')).fetchone()[0] == 'Bank'


def test_heal_is_idempotent_noop_when_complete(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path/'complete.db'}")
    # Full, current schema -> guard should add nothing.
    User.__table__.create(eng)
    Account.__table__.create(eng)
    assert _heal_missing_columns(eng, [User, Account]) == []
    # Running again is still a no-op.
    assert _heal_missing_columns(eng, [User, Account]) == []


def test_heal_skips_absent_table(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path/'empty.db'}")
    # No tables at all -> nothing to heal, never raises.
    assert _heal_missing_columns(eng, [User, Account]) == []


def test_heal_adds_explanation_source_to_transaction(tmp_path):
    """The client-explain feature added a NOT-NULL `explanation_source` column via
    migration only; on a create_all()-built prod DB it would be missing and 500
    every Transaction query. The guard must heal it."""
    eng = create_engine(f"sqlite:///{tmp_path/'legacy_txn.db'}")
    with eng.begin() as c:
        # Legacy 'transaction' table predating explanation_source.
        c.execute(text(
            'CREATE TABLE "transaction" (id INTEGER PRIMARY KEY, date DATETIME, '
            'description VARCHAR(200), amount FLOAT, user_id INTEGER, '
            'file_id INTEGER, explanation VARCHAR(500))'))
        c.execute(text(
            'INSERT INTO "transaction" (id, date, description, amount, user_id, explanation) '
            "VALUES (1, '2026-03-01', 'Coffee', -4.5, 1, '')"))

    assert 'explanation_source' not in {
        c['name'] for c in inspect(eng).get_columns('transaction')}

    added = _heal_missing_columns(eng, [Transaction])

    txn_cols = {c['name'] for c in inspect(eng).get_columns('transaction')}
    assert {col.name for col in Transaction.__table__.columns} <= txn_cols
    assert ('transaction', 'explanation_source') in added
    # Existing row preserved; new column added NULLable so the legacy row survives.
    with eng.begin() as c:
        assert c.execute(
            text('SELECT description FROM "transaction" WHERE id=1')).fetchone()[0] == 'Coffee'
