"""login_attempt — the failed-sign-in ledger behind the login throttle.

Revision ID: d5e1f7a9c2b4
Revises: c4e8a2b7d9f1
Create Date: 2026-10-06

The deploy builds its schema with create_all(), which creates this table on
its own; this migration is for any environment that runs `flask db upgrade`.
Keys are SHA-256 hashes — no e-mail address or IP is stored in clear.
"""
from alembic import op
import sqlalchemy as sa


revision = 'd5e1f7a9c2b4'
down_revision = 'c4e8a2b7d9f1'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'login_attempt',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('key_hash', sa.String(length=64), nullable=False),
        sa.Column('kind', sa.String(length=8), nullable=False),
        sa.Column('attempted_at', sa.DateTime(), nullable=False),
        sa.Column('succeeded', sa.Boolean(), nullable=False),
    )
    op.create_index('ix_login_attempt_key_hash', 'login_attempt', ['key_hash'])
    op.create_index('ix_login_attempt_attempted_at', 'login_attempt', ['attempted_at'])


def downgrade():
    op.drop_index('ix_login_attempt_attempted_at', table_name='login_attempt')
    op.drop_index('ix_login_attempt_key_hash', table_name='login_attempt')
    op.drop_table('login_attempt')
