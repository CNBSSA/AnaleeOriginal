"""The Practice Club — SSO consumer entry for Analee.

A Club Member already signed in at the hub (`clubhub`) opens Analee. The hub
mints a short-lived RS256 access token (audience ``analee``) and redirects
here: ``GET /sso/enter/?token=…``. This view validates the token, maps the
member to their own Analee user (JIT-provisioned on first entry, chart
included), logs them in, and lands them on their dashboard.

Ships DARK behind ``CLUB_ENABLED`` (default off → this view 404s; the existing
login and every direct flow are unaffected — Iron Rule).

Analee tenancy: PER-USER. There is no client/company concept here, so a
``target_client_company_id`` claim is deliberately ignored — the member always
enters their own workspace.

Security invariants (mirror the other five consumers):
- Analee stores only the hub **public** key; tokens are short-lived; the
  algorithm is pinned to RS256 in ``jwt_util``.
- The SSO user gets a random password it never learns, so the email/password
  login path can never authenticate as it.
"""
from __future__ import annotations

import logging
import os
import secrets

from flask import Blueprint, abort, redirect, request, session, url_for
from flask_login import login_user

from club_sso import jwt_util
from club_sso.models import ClubMemberLink
from models import User, db

bp = Blueprint("club_sso", __name__, url_prefix="/sso")
logger = logging.getLogger(__name__)

AUDIENCE = "analee"


def _enabled() -> bool:
    return (os.environ.get("CLUB_ENABLED", "") or "").strip().lower() in (
        "1", "true", "yes")


def _resolve_user(member_id, seat_id) -> User:
    """Map a hub (member, seat) to an Analee user, JIT-provisioning one (with
    their chart of accounts) on first SSO entry."""
    link = ClubMemberLink.query.filter_by(
        hub_member_id=str(member_id), seat_id=str(seat_id)).first()
    if link is not None:
        return link.user

    email = f"club+{member_id}.{seat_id}@sso.theaccountants.local"
    user = User.query.filter_by(email=email).first()
    if user is None:
        user = User(username=email[:64], email=email,
                    subscription_status="active")
        user.set_password(secrets.token_urlsafe(32))  # random, never shared
        db.session.add(user)
        db.session.flush()  # assign user.id
        try:
            # Provision their chart via the frozen service (called, not changed).
            User.create_default_accounts(user.id)
        except Exception:  # noqa: BLE001 — a chart hiccup must not block entry
            logger.exception(
                "club_sso: chart provisioning failed for user %s "
                "(login proceeds; seed-charts can heal later)", user.id)

    db.session.add(ClubMemberLink(
        hub_member_id=str(member_id), seat_id=str(seat_id), user_id=user.id))
    db.session.commit()
    return user


@bp.route("/enter/")
def enter():
    """Validate a hub token and drop the member into their Analee workspace."""
    if not _enabled():
        abort(404)

    pub = (os.environ.get("HUB_JWT_PUBLIC_KEY", "") or "").strip()
    if not pub:
        abort(404)  # not configured → behave as if dark

    token = request.args.get("token", "")
    try:
        payload = jwt_util.verify_rs256(
            token, pub, audience=AUDIENCE,
            issuer=os.environ.get("HUB_ISSUER", "the-accountants-hub"))
    except jwt_util.JWTError:
        return "invalid token", 401

    member_id = payload.get("member_id")
    seat_id = payload.get("seat_id")
    if member_id is None or seat_id is None:
        return "incomplete token", 401

    # P4 (One-Login Practice Layer, Festus 2026-07-19): a practice
    # accountant's Club identity IS their Analee identity. When the
    # hub-verified email (a signed claim; hub emails are mailbox-verified at
    # §25 onboarding) matches a real accountant account that carries a
    # PracticeLink, log them into THAT account and land on My Clients — one
    # identity across the estate. Ordinary members keep the existing
    # per-member alias path unchanged; alias-domain emails are excluded so a
    # hidden workspace/SSO identity can never be entered this way.
    try:
        import practice_layer
        email = (payload.get("email") or "").strip().lower()
        if practice_layer.enabled() and email and "@" in email \
                and not email.endswith(".theaccountants.local"):
            from models import PracticeLink
            real = User.query.filter(
                db.func.lower(User.email) == email).first()
            if (real is not None and not real.is_deleted
                    and PracticeLink.query.filter_by(
                        accountant_user_id=real.id).first() is not None):
                login_user(real)
                session["club_session"] = True
                session["club_member_id"] = str(member_id)
                return redirect("/practice")
    except Exception:  # noqa: BLE001 — never let P4 break plain SSO entry
        logger.exception("club_sso: practice one-login check failed "
                         "(falling back to member alias entry)")

    user = _resolve_user(member_id, seat_id)
    login_user(user)
    session["club_session"] = True
    session["club_member_id"] = str(member_id)
    return redirect(url_for("main.dashboard"))
