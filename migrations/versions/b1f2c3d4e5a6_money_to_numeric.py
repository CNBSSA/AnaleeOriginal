"""Convert currency columns from Float to Numeric(18, 2)

Money must be exact to the cent (IFRS). The ledger/currency columns were created
as Float (double precision), which cannot represent decimal currency exactly.
Convert them to NUMERIC(18, 2). Non-currency metric/ratio columns (risk values,
alert thresholds) are intentionally left as Float.

The application also self-heals existing PostgreSQL databases at boot (see the
money-precision guard in app.create_app), so this migration is for environments
that run `flask db upgrade`.

Revision ID: b1f2c3d4e5a6
Revises: ad7c0f1d9b2a
Create Date: 2026-06-25
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'b1f2c3d4e5a6'
down_revision = 'ad7c0f1d9b2a'
branch_labels = None
depends_on = None

# (table, column, nullable)
_COLS = [
    ('transaction', 'amount', False),
    ('historical_data', 'amount', False),
    ('financial_goal', 'target_amount', False),
    ('financial_goal', 'current_amount', True),
]


def upgrade():
    bind = op.get_bind()
    is_pg = bind.dialect.name == 'postgresql'
    for table, col, nullable in _COLS:
        kwargs = dict(type_=sa.Numeric(18, 2), existing_type=sa.Float(),
                      existing_nullable=nullable)
        if is_pg:
            kwargs['postgresql_using'] = f'{col}::numeric(18,2)'
        with op.batch_alter_table(table) as batch:
            batch.alter_column(col, **kwargs)


def downgrade():
    for table, col, nullable in _COLS:
        with op.batch_alter_table(table) as batch:
            batch.alter_column(col, type_=sa.Float(),
                               existing_type=sa.Numeric(18, 2),
                               existing_nullable=nullable)
