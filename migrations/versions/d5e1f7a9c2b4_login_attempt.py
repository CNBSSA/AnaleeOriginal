"""login_attempt — documented in the baseline; chain head for Alembic.

Revision ID: d5e1f7a9c2b4
Revises: f8e7d6c5b4a3
Create Date: 2026-10-06

The ``login_attempt`` table is created by the baseline migration
(``f8e7d6c5b4a3``). This revision remains so environments that already
stamped ``d5e1f7a9c2b4`` stay compatible and the additive-table migration
test keeps a named file beside ``create_all()`` boot behaviour.
"""
from alembic import op


revision = 'd5e1f7a9c2b4'
down_revision = 'f8e7d6c5b4a3'
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
