"""Baseline schema — current Analee models on a fresh database.

Revision ID: f8e7d6c5b4a3
Revises:
Create Date: 2026-10-11

Production deploys still use boot-time ``db.create_all()``; this revision is
the production-equivalent path exercised in CI: ``flask db upgrade`` against
PostgreSQL on an empty database.

Earlier incremental revisions lived under ``migrations/_legacy_versions/`` and
were never runnable from empty — they assumed tables created before Alembic.
"""
from alembic import op


revision = 'f8e7d6c5b4a3'
down_revision = None
branch_labels = None
depends_on = None


def _all_metadata():
    import models  # noqa: F401 — register core tables on db.metadata
    import club_sso.models  # noqa: F401 — club_member_link
    from models import db
    return db.metadata


def upgrade():
    bind = op.get_bind()
    _all_metadata().create_all(bind=bind, checkfirst=True)


def downgrade():
    bind = op.get_bind()
    _all_metadata().drop_all(bind=bind)
