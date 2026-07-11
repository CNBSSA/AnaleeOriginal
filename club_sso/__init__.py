"""The Practice Club — SSO consumer for Analee (SEALED MODULE).

Scoped unfreeze recorded in CLAUDE.md (Festus, 2026-07-10): AnaleeOriginal was
re-opened for THIS module only; everything else stays frozen and the repo is
re-frozen with this module inside.

Design constraints (why this module looks the way it does):
- ONE touchpoint: app.py makes a single `club_sso.register(app)` call. No other
  existing file changes. Removing that call removes the feature entirely.
- DARK by default: every route 404s unless CLUB_ENABLED is set. With the flag
  off, Analee runs byte-for-byte as before.
- FAIL-SOFT boot: register() can never raise — a failure here can never stop
  Analee from starting (the app Festus fears breaking most).
- No migrations, no Procfile change: this repo deploys with db.create_all()
  (see app.py), so the one new table is ensured idempotently at register time
  and the walkthrough seed self-arms from CLUB_WALKTHROUGH_SEED=1.

Analee tenancy note: unlike BooksXperts/TrustEasyGo (accountant → client) the
workspace here is PER-USER — a Club member maps to their own Analee user and
lands on their own dashboard. A `target_client_company_id` in the token is
deliberately ignored.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def register(app):
    """Single entry point called once from app.py. Never raises."""
    try:
        from club_sso.routes import bp
        from club_sso.seed import register_cli, run_seed
        from club_sso.models import ClubMemberLink
        from models import db

        app.register_blueprint(bp)
        register_cli(app)

        # This repo deploys with create_all(), not migrations; ensure our one
        # (additive) table exists. checkfirst → idempotent, never ALTERs.
        ClubMemberLink.__table__.create(bind=db.engine, checkfirst=True)

        # Walkthrough seed arms itself from the env var — no shell, no command
        # edits (same contract as the other five products).
        if os.environ.get("CLUB_WALKTHROUGH_SEED") == "1":
            run_seed()
    except Exception as exc:  # noqa: BLE001 — boot must never be blocked
        logger.error("club_sso registration skipped after error: %s", exc)
