"""Flask app entry for ``flask db upgrade`` (migration gate / CI).

Set ``ANALEE_SKIP_BOOT_SCHEMA=1`` before import so Alembic builds schema via
migrations instead of boot-time ``create_all()``.
"""
from app import create_app

app = create_app()
