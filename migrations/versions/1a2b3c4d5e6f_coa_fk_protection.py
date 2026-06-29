"""coa_fk_protection: RESTRICT transaction→account, SET NULL upload→account

Revision ID: 1a2b3c4d5e6f
Revises: f6765e4cf063
Create Date: 2026-06-29

Changes:
  - transaction.account_id FK: CASCADE → RESTRICT
      A categorised transaction now blocks account deletion at the DB level,
      matching the application-layer guard added to delete_account().
  - bank_statement_upload.account_id: not-null CASCADE → nullable SET NULL
      Deleting an account no longer destroys the upload record; the account
      reference is cleared to NULL instead.
"""
from alembic import op
import sqlalchemy as sa


revision = '1a2b3c4d5e6f'
down_revision = 'f6765e4cf063'
branch_labels = None
depends_on = None


def upgrade():
    # ── transaction.account_id: CASCADE → RESTRICT ────────────────────────
    with op.batch_alter_table('transaction', schema=None) as batch_op:
        batch_op.drop_constraint('transaction_account_id_fkey', type_='foreignkey')
        batch_op.create_foreign_key(
            'transaction_account_id_fkey',
            'account', ['account_id'], ['id'],
            ondelete='RESTRICT',
        )

    # ── bank_statement_upload.account_id: not-null CASCADE → nullable SET NULL
    with op.batch_alter_table('bank_statement_upload', schema=None) as batch_op:
        batch_op.drop_constraint('bank_statement_upload_account_id_fkey', type_='foreignkey')
        batch_op.alter_column(
            'account_id',
            existing_type=sa.Integer(),
            nullable=True,
        )
        batch_op.create_foreign_key(
            'bank_statement_upload_account_id_fkey',
            'account', ['account_id'], ['id'],
            ondelete='SET NULL',
        )


def downgrade():
    # ── bank_statement_upload.account_id: restore not-null CASCADE ─────────
    with op.batch_alter_table('bank_statement_upload', schema=None) as batch_op:
        batch_op.drop_constraint('bank_statement_upload_account_id_fkey', type_='foreignkey')
        batch_op.alter_column(
            'account_id',
            existing_type=sa.Integer(),
            nullable=False,
        )
        batch_op.create_foreign_key(
            'bank_statement_upload_account_id_fkey',
            'account', ['account_id'], ['id'],
            ondelete='CASCADE',
        )

    # ── transaction.account_id: restore CASCADE ────────────────────────────
    with op.batch_alter_table('transaction', schema=None) as batch_op:
        batch_op.drop_constraint('transaction_account_id_fkey', type_='foreignkey')
        batch_op.create_foreign_key(
            'transaction_account_id_fkey',
            'account', ['account_id'], ['id'],
            ondelete='CASCADE',
        )
