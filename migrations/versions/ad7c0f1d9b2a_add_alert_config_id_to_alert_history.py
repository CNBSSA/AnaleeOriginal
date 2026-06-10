"""Add alert_config_id FK to alert_history

The model `AlertHistory.alert_config_id` (FK -> alert_configuration.id, CASCADE)
was added in code, but the alert tables are created by db.create_all(), not by
the migration chain. On a fresh database create_all() builds the column; on an
EXISTING database create_all() never ALTERs the table, so the column is missing
and alert generation (which writes alert_config_id) fails.

This migration adds the column to existing databases. It is:
  * idempotent  — no-op if the table is absent (create_all will build it with
    the column already present) or if the column already exists;
  * nullable in the DB — the table may already hold historical rows that have no
    config to point at, and a NOT NULL column with no default cannot be added to
    a populated table. The model keeps nullable=False, so every NEW row written
    by the app supplies the value; only legacy rows may be NULL.

Revision ID: ad7c0f1d9b2a
Revises: 5a92dda42cc9
Create Date: 2026-06-10
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'ad7c0f1d9b2a'
down_revision = '5a92dda42cc9'
branch_labels = None
depends_on = None

_TABLE = 'alert_history'
_COL = 'alert_config_id'
_FK = 'fk_alert_history_alert_config_id'


def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if _TABLE not in insp.get_table_names():
        return  # create_all() will build it with the column already present
    cols = [c['name'] for c in insp.get_columns(_TABLE)]
    if _COL in cols:
        return  # already present (e.g. create_all on a fresh DB)
    with op.batch_alter_table(_TABLE) as batch:
        batch.add_column(sa.Column(_COL, sa.Integer(), nullable=True))
        batch.create_foreign_key(
            _FK, 'alert_configuration', [_COL], ['id'], ondelete='CASCADE'
        )


def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if _TABLE not in insp.get_table_names():
        return
    cols = [c['name'] for c in insp.get_columns(_TABLE)]
    if _COL not in cols:
        return
    # Dropping the column removes its FK constraint on both PostgreSQL (auto) and
    # SQLite (table rebuilt), so an explicit drop_constraint isn't needed.
    with op.batch_alter_table(_TABLE) as batch:
        batch.drop_column(_COL)
