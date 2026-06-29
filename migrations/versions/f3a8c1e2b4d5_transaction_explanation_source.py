"""Revision ID: f3a8c1e2b4d5
Revises: e8a1c4b2d9f0
Create Date: 2026-06-28
"""
from alembic import op
import sqlalchemy as sa


revision = 'f3a8c1e2b4d5'
down_revision = 'e8a1c4b2d9f0'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('transaction', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('explanation_source', sa.String(length=20), nullable=False, server_default=''),
        )


def downgrade():
    with op.batch_alter_table('transaction', schema=None) as batch_op:
        batch_op.drop_column('explanation_source')
