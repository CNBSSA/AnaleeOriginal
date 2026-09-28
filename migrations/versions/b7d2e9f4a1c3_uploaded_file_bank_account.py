"""uploaded_file.bank_account_id — the bank a statement belongs to.

Revision ID: b7d2e9f4a1c3
Revises: f3a8c1e2b4d5
Create Date: 2026-09-28

The deploy builds its schema with create_all() plus the boot-time column heal
in app.py, which adds this column on its own; this migration is for any
environment that runs `flask db upgrade`.
"""
from alembic import op
import sqlalchemy as sa


revision = 'b7d2e9f4a1c3'
down_revision = 'f3a8c1e2b4d5'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('uploaded_file', schema=None) as batch_op:
        batch_op.add_column(sa.Column('bank_account_id', sa.Integer(), nullable=True))


def downgrade():
    with op.batch_alter_table('uploaded_file', schema=None) as batch_op:
        batch_op.drop_column('bank_account_id')
