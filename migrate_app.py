"""Flask app entry for ``flask db upgrade`` (migration gate / CI).

Set ``ANALEE_SKIP_BOOT_SCHEMA=1`` before import so Alembic builds schema via
migrations instead of boot-time ``create_all()``.

GitHub Actions checks out into ``.../AnaleeOriginal/AnaleeOriginal``; the Flask
CLI may import this file as ``AnaleeOriginal.migrate_app``, so the repo root
must be on ``sys.path`` before ``from app import create_app``.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app import create_app

app = create_app()
