"""trial_balance_approval — admin approval before a TB goes to THE ACCOUNTANTS.

Revision ID: c4e8a2b7d9f1
Revises: b7d2e9f4a1c3
Create Date: 2026-09-28

The deploy builds its schema with create_all(), which creates this table on
its own; this migration is for any environment that runs `flask db upgrade`.
"""
from alembic import op
import sqlalchemy as sa


revision = 'c4e8a2b7d9f1'
down_revision = 'b7d2e9f4a1c3'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'trial_balance_approval',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('user.id', ondelete='CASCADE'),
                  nullable=False, index=True),
        sa.Column('period_start', sa.Date()),
        sa.Column('period_end', sa.Date(), nullable=False, index=True),
        sa.Column('tb_fingerprint', sa.String(64), nullable=False),
        sa.Column('row_count', sa.Integer()),
        sa.Column('total_debits', sa.Float()),
        sa.Column('total_credits', sa.Float()),
        sa.Column('status', sa.String(16), nullable=False, server_default='requested'),
        sa.Column('requested_by', sa.Integer(), sa.ForeignKey('user.id', ondelete='SET NULL')),
        sa.Column('requested_at', sa.DateTime()),
        sa.Column('decided_by', sa.Integer(), sa.ForeignKey('user.id', ondelete='SET NULL')),
        sa.Column('decided_at', sa.DateTime()),
        sa.Column('note', sa.String(500)),
    )


def downgrade():
    op.drop_table('trial_balance_approval')
