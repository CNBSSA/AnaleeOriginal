"""Walkthrough seed for Analee — The Practice Club demo member (sentinel 900001).

Same env contract as the other five products: the seed runs ONLY when
``CLUB_WALKTHROUGH_SEED=1`` is set on that environment (staging only — never
set it in production). It is invoked automatically by ``club_sso.register()``
at boot, and also available as ``flask seed-club-demo`` for parity.

Idempotent: safe to run on every boot.
"""
from __future__ import annotations

import logging
import os
import secrets

logger = logging.getLogger(__name__)

# Walkthrough sentinels — must match clubhub/hub/club_demo.py.
DEMO_MEMBER_ID = "900001"
DEMO_SEAT_ID = "900001"
DEMO_EMAIL = f"club+{DEMO_MEMBER_ID}.{DEMO_SEAT_ID}@sso.theaccountants.local"


def run_seed():
    """Create the demo member's Analee user + link. No-ops unless armed."""
    if os.environ.get("CLUB_WALKTHROUGH_SEED") != "1":
        print("seed-club-demo skipped (CLUB_WALKTHROUGH_SEED != 1 — staging-only seed).")
        return

    from club_sso.models import ClubMemberLink
    from models import User, db

    ClubMemberLink.__table__.create(bind=db.engine, checkfirst=True)

    user = User.query.filter_by(email=DEMO_EMAIL).first()
    if user is None:
        user = User(username=DEMO_EMAIL[:64], email=DEMO_EMAIL,
                    subscription_status="active")
        user.set_password(secrets.token_urlsafe(32))
        db.session.add(user)
        db.session.flush()
        try:
            User.create_default_accounts(user.id)
        except Exception:  # noqa: BLE001
            logger.exception("seed-club-demo: chart provisioning failed "
                             "(user still created; seed-charts can heal later)")

    if not ClubMemberLink.query.filter_by(
            hub_member_id=DEMO_MEMBER_ID, seat_id=DEMO_SEAT_ID).first():
        db.session.add(ClubMemberLink(
            hub_member_id=DEMO_MEMBER_ID, seat_id=DEMO_SEAT_ID, user_id=user.id))

    db.session.commit()
    print(f"seed-club-demo: demo member {DEMO_MEMBER_ID} ready.")


def register_cli(app):
    @app.cli.command("seed-club-demo")
    def seed_club_demo():
        """Seed Analee's slice of the Club walkthrough (env-guarded)."""
        run_seed()
