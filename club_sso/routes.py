"""The Practice Club — SSO consumer entry for Analee.

A Club Member already signed in at the hub (`clubhub`) opens Analee. The hub
mints a short-lived RS256 access token (audience ``analee``) and redirects
here: ``GET /sso/enter/?token=…``. This view validates the token, maps the
member to their own Analee user (JIT-provisioned on first entry, chart
included), logs them in, and lands them on their dashboard.

Ships DARK behind ``CLUB_ENABLED`` (default off → this view 404s; the existing
login and every direct flow are unaffected — Iron Rule).

Analee tenancy: PER-USER for the member alias, but a hub ``target_client_ref``
(Phase C workspace ref, e.g. ``club-7-42``) opens that client's dedicated
workspace when it belongs to the signed-in member.

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


def _trusted_public_keys() -> list:
    """The hub public keys this consumer trusts.

    Zero-downtime rotation (mirrors clubhub #141): during a hub key overlap the
    multi-key var may hold several concatenated PEM blocks — the new key and the
    previous one — and a token signed by EITHER is accepted. The multi-key var is
    read as ``HUB_JWT_PUBLIC_KEYS`` (matching this repo's existing
    ``HUB_JWT_PUBLIC_KEY``), and also as ``CLUB_JWT_PUBLIC_KEYS`` for parity with
    the hub's env naming. When neither is set the trusted set is exactly
    ``[HUB_JWT_PUBLIC_KEY]`` — byte-identical to before this change (dark until
    the multi-key var is set). Empty → ``[]`` → the caller stays dark (404)."""
    multi = (os.environ.get("HUB_JWT_PUBLIC_KEYS")
             or os.environ.get("CLUB_JWT_PUBLIC_KEYS") or "").strip()
    if multi:
        keys = jwt_util.split_pem_blocks(multi)
        if keys:
            return keys
    single = (os.environ.get("HUB_JWT_PUBLIC_KEY", "") or "").strip()
    return [single] if single else []


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


def _enter_member_workspace(member_id, client_ref: str):
    """Open a Club-provisioned client workspace when the ref is scoped to member."""
    from provisioning import _is_workspace_email, workspace_email

    ref = (client_ref or "").strip()
    prefix = f"club-{member_id}-"
    if not ref.startswith(prefix):
        return None
    email = workspace_email(ref)
    if not email:
        return None
    ws = User.query.filter(db.func.lower(User.email) == email.lower()).first()
    if (ws is None or ws.is_deleted
            or not _is_workspace_email(ws.email)
            or ws.subscription_status != "active"):
        return None
    login_user(ws)
    session["club_session"] = True
    session["club_member_id"] = str(member_id)
    session["workspace_session"] = True
    session["workspace_email"] = ws.email
    return redirect(url_for("main.dashboard"))


@bp.route("/enter/")
def enter():
    """Validate a hub token and drop the member into their Analee workspace."""
    if not _enabled():
        abort(404)

    keys = _trusted_public_keys()
    if not keys:
        abort(404)  # not configured → behave as if dark

    token = request.args.get("token", "")
    try:
        payload = jwt_util.verify_rs256(
            token, keys, audience=AUDIENCE,
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

    client_ref = (payload.get("target_client_ref")
                  or request.args.get("client") or "").strip()
    if client_ref:
        ws_redirect = _enter_member_workspace(member_id, client_ref)
        if ws_redirect is not None:
            return ws_redirect

    user = _resolve_user(member_id, seat_id)
    login_user(user)
    session["club_session"] = True
    session["club_member_id"] = str(member_id)
    return redirect(url_for("main.dashboard"))
