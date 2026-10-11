"""add logo to company_settings (Accountants Club branding prerequisite)

Adds practice-branding columns to company_settings so Analee can carry the
member's firm logo for white-label reports + the Club attribution/watermark
(Accountants Club §13/§20 — the named P5 prerequisite for Analee).

Revision ID: c1f5b2a4d8e9
Revises: ad7c0f1d9b2a
Create Date: 2026-06-19
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c1f5b2a4d8e9'
down_revision = 'ad7c0f1d9b2a'
branch_labels = None
depends_on = None


def upgrade():
    # batch mode for SQLite compatibility (ALTER ADD COLUMN limitations).
    with op.batch_alter_table('company_settings', schema=None) as batch_op:
        batch_op.add_column(sa.Column('logo', sa.LargeBinary(), nullable=True))
        batch_op.add_column(sa.Column('logo_type', sa.String(length=50), nullable=True))


def downgrade():
    with op.batch_alter_table('company_settings', schema=None) as batch_op:
        batch_op.drop_column('logo_type')
        batch_op.drop_column('logo')
