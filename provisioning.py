"""Cross-product Analee provisioning (dark, Slice 5b — Festus 2026-07-13).

THE ACCOUNTANTS and Analee are priced together: buying THE ACCOUNTANTS as a
standalone grants **full** Analee access (policy D-F). Analee learns "this
person is an Accountants subscriber" via a **server-to-server** provisioning
call from THE ACCOUNTANTS that activates (or deactivates) the matching Analee
user **by email**.

Ships **DARK** + **fail-closed**:
- ``ANALEE_PROVISIONING_ENABLED`` (default off) gates the endpoint entirely →
  404 while off, so the surface does not exist until Festus turns it on.
- ``ANALEE_PROVISIONING_SECRET`` must be set; a missing secret → 503; a
  wrong/absent bearer token → 401. Compared in constant time.

Reversible: ``entitled=false`` deactivates. **No schema change** — it reuses the
existing ``subscription_status`` (active/inactive), i.e. the same
``is_active`` gate Analee already uses. ``is_deleted`` is never touched.

IDENTITY = **email** (the natural cross-product key). *Flagged for Festus:*
confirm email-matching, or switch to an explicit account-link, before enabling.

CLIENT WORKSPACES (scoped re-open, Festus 2026-07-15 — "a companion of the
accountants"): THE ACCOUNTANTS orchestrates one Analee workspace **per firm
client** so bank statements can be analysed per client and the trial balance
flows back to the right client. Same seam, same flags, same fail-closed bearer:

- ``POST /api/provisioning/analee/workspace`` — idempotently create (or fetch)
  a client workspace: a dedicated Analee ``User`` under a deterministic alias
  email (``client+<ref>@ws.theaccountants.local``) with a random password no
  one ever learns, plus ``CompanySettings`` named after the client and the
  entity-correct chart of accounts via the frozen chart service (called, never
  changed). **No schema change** — the one-company-per-user model is reused
  as one-workspace-per-client.
- ``POST /api/provisioning/analee/workspace/login-link`` — mint a short-TTL
  (default 90 s) signed login path for a workspace, so the accountant clicks
  straight into the client's Analee. Signed with ``itsdangerous`` off the same
  ``ANALEE_PROVISIONING_SECRET`` (purpose-salted); refuses to mint for any
  non-workspace account, so it can never become an account-takeover vector.
- ``GET /workspace/enter?token=…`` — the browser lands here; the token is
  verified (TTL-bounded, not single-use — the window is seconds and only
  workspace aliases are eligible) and the accountant is logged into the
  client workspace.
"""
import hmac
import logging
import os
import re
import secrets

from flask import Blueprint, current_app, flash, jsonify, redirect, request, session, url_for

logger = logging.getLogger(__name__)

provisioning = Blueprint("provisioning", __name__)


def enabled():
    """True when the provisioning endpoint is switched on for this environment."""
    return os.environ.get("ANALEE_PROVISIONING_ENABLED", "False") == "True"


def _secret():
    return os.environ.get("ANALEE_PROVISIONING_SECRET", "") or ""


def _bearer_ok(header_value):
    """Constant-time check of an ``Authorization: Bearer <secret>`` header."""
    secret = _secret()
    if not secret:
        return None  # signals "not configured" → 503
    prefix = "Bearer "
    token = header_value[len(prefix):] if (header_value or "").startswith(prefix) else ""
    return hmac.compare_digest(token, secret)


def provision_by_email(email, entitled):
    """Activate/deactivate the Analee user matching ``email``.

    Returns a small result dict. Never raises for a missing user — returns
    ``found=False`` so the caller can log without leaking existence to the wire
    via status codes. Reuses ``subscription_status`` (no migration); leaves
    ``is_deleted`` untouched.
    """
    from models import db, User

    user = User.query.filter(db.func.lower(User.email) == (email or "").strip().lower()).first()
    if user is None:
        logger.info("provisioning: no Analee user for email (entitled=%s)", entitled)
        return {"found": False, "entitled": bool(entitled)}
    user.subscription_status = "active" if entitled else "inactive"
    db.session.commit()
    logger.info("provisioning: user %s set subscription_status=%s",
                user.id, user.subscription_status)
    return {"found": True, "entitled": bool(entitled),
            "subscription_status": user.subscription_status}


@provisioning.route("/api/provisioning/analee", methods=["POST"])
def provision_analee():
    """S2S entry point for THE ACCOUNTANTS to grant/revoke Analee access."""
    if not enabled():
        # Dark: the surface does not exist until explicitly enabled.
        return jsonify({"error": "not found"}), 404
    ok = _bearer_ok(request.headers.get("Authorization", ""))
    if ok is None:
        return jsonify({"error": "provisioning not configured"}), 503
    if not ok:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip()
    if not email:
        return jsonify({"error": "email is required"}), 400
    entitled = bool(data.get("entitled", True))
    result = provision_by_email(email, entitled)
    return jsonify(result), 200


# --------------------------------------------------------------------------
# Client workspaces (scoped re-open, Festus 2026-07-15)
# --------------------------------------------------------------------------

WORKSPACE_EMAIL_DOMAIN = "ws.theaccountants.local"
_WORKSPACE_LOGIN_SALT = "analee-workspace-login-v1"
DEFAULT_LINK_TTL_SECONDS = 90


def _link_ttl():
    try:
        return int(os.environ.get("ANALEE_WORKSPACE_LINK_TTL",
                                  DEFAULT_LINK_TTL_SECONDS))
    except (TypeError, ValueError):
        return DEFAULT_LINK_TTL_SECONDS


def workspace_email(client_ref):
    """Deterministic alias email for a firm client — the idempotency key.

    ``client_ref`` is THE ACCOUNTANTS' stable identifier for the client (it
    owns the mapping). Sanitised to a safe local-part; empty/oversized refs
    are rejected so a malformed caller can't mint junk identities."""
    safe = re.sub(r"[^a-z0-9]+", "-", (client_ref or "").strip().lower()).strip("-")
    if not safe or len(safe) > 60:
        return None
    return f"client+{safe}@{WORKSPACE_EMAIL_DOMAIN}"


def _is_workspace_email(email):
    return (email or "").lower().endswith("@" + WORKSPACE_EMAIL_DOMAIN)


def ensure_workspace(client_ref, client_name, entity_name=None):
    """Idempotently create (or fetch) the Analee workspace for a firm client.

    Creates a dedicated ``User`` (random password nobody learns — the only way
    in is the signed login link), ``CompanySettings`` carrying the client's
    real name, and the entity-correct chart via the frozen chart service
    (called, never modified). Re-ensuring an existing workspace refreshes the
    company name and reactivates it if it had been revoked. No schema change.
    """
    from models import db, Entity, User, CompanySettings

    email = workspace_email(client_ref)
    if email is None:
        return {"error": "client_ref is required (letters/digits)"}

    user = User.query.filter(
        db.func.lower(User.email) == email).first()
    if user is not None:
        # Idempotent re-ensure: reactivate + keep the client name in step.
        user.subscription_status = "active"
        settings = CompanySettings.query.filter_by(user_id=user.id).first()
        if settings is not None and client_name and settings.company_name != client_name:
            settings.company_name = client_name
        db.session.commit()
        return {"created": False, "workspace_user_id": user.id, "email": email,
                "company": settings.company_name if settings else None}

    user = User(username=email[:64], email=email, subscription_status="active")
    user.set_password(secrets.token_urlsafe(32))  # random, never shared
    db.session.add(user)
    db.session.flush()  # assign user.id

    settings = CompanySettings(
        user_id=user.id,
        company_name=(client_name or "").strip() or "Client Workspace",
        financial_year_end=2,  # SA default (Feb year-end); editable in-app
    )
    db.session.add(settings)
    db.session.flush()

    chart_provisioned = True
    entity_used = None
    try:
        # Chart via the FROZEN service — called, never changed (same pattern
        # as club_sso JIT provisioning). Unknown entity names fall back to the
        # service default (Private Company) rather than failing the ensure.
        from services.chart_of_accounts import set_entity_for_user
        entity = None
        if entity_name:
            entity = Entity.query.filter(
                db.func.lower(Entity.name) == entity_name.strip().lower()).first()
        if entity is not None:
            set_entity_for_user(user.id, entity.id)
            entity_used = entity.name
        else:
            User.create_default_accounts(user.id)
            entity_used = "Private Company"
    except Exception:  # noqa: BLE001 — a chart hiccup must not lose the workspace
        chart_provisioned = False
        logger.exception("workspace provisioning: chart failed for user %s "
                         "(workspace kept; re-ensure heals)", user.id)
    db.session.commit()
    logger.info("workspace provisioning: created workspace user %s (%s)",
                user.id, email)
    return {"created": True, "workspace_user_id": user.id, "email": email,
            "company": settings.company_name, "entity": entity_used,
            "chart_provisioned": chart_provisioned}


def _login_serializer():
    from itsdangerous import URLSafeTimedSerializer
    return URLSafeTimedSerializer(_secret(), salt=_WORKSPACE_LOGIN_SALT)


@provisioning.route("/api/provisioning/analee/workspace", methods=["POST"])
def workspace_ensure():
    """S2S: THE ACCOUNTANTS ensures a client workspace exists."""
    if not enabled():
        return jsonify({"error": "not found"}), 404
    ok = _bearer_ok(request.headers.get("Authorization", ""))
    if ok is None:
        return jsonify({"error": "provisioning not configured"}), 503
    if not ok:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    result = ensure_workspace(
        data.get("client_ref"), data.get("client_name"),
        data.get("entity_name"))
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result), 200


@provisioning.route("/api/provisioning/analee/workspace/login-link",
                    methods=["POST"])
def workspace_login_link():
    """S2S: mint a short-TTL signed login path into a client workspace.

    Only workspace-alias accounts are eligible — a regular user's email is
    refused outright, so this can never authenticate as a human account."""
    if not enabled():
        return jsonify({"error": "not found"}), 404
    ok = _bearer_ok(request.headers.get("Authorization", ""))
    if ok is None:
        return jsonify({"error": "provisioning not configured"}), 503
    if not ok:
        return jsonify({"error": "unauthorized"}), 401
    from models import db, User

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if not email and data.get("client_ref"):
        email = workspace_email(data.get("client_ref")) or ""
    if not _is_workspace_email(email):
        return jsonify({"error": "not a workspace account"}), 400
    user = User.query.filter(db.func.lower(User.email) == email).first()
    if user is None or user.is_deleted or user.subscription_status != "active":
        return jsonify({"found": False}), 200
    token = _login_serializer().dumps({"uid": user.id})
    return jsonify({"found": True,
                    "url_path": f"/workspace/enter?token={token}",
                    "expires_in": _link_ttl()}), 200


def practice_firm_ref(raw_ref):
    """Sanitise THE ACCOUNTANTS' firm ref (e.g. ``acc-7``) the same way client
    refs are sanitised, so prefix matching against workspace alias emails is
    exact. Returns None for empty/oversized junk."""
    safe = re.sub(r"[^a-z0-9]+", "-", (raw_ref or "").strip().lower()).strip("-")
    if not safe or len(safe) > 40:
        return None
    return safe


def ensure_practice_link(accountant_email, firm_ref, firm_name=None):
    """Idempotently bind an accountant's own Analee account to their firm.

    Creates the accountant ``User`` if absent (random password — they set their
    own via the normal password reset; the alias-domain is refused so this can
    never bind a hidden workspace identity as an accountant). One firm per
    accountant (P1); re-ensuring updates firm name and re-points the ref."""
    from models import db, User, PracticeLink

    email = (accountant_email or "").strip().lower()
    if not email or "@" not in email or _is_workspace_email(email):
        return {"error": "a real accountant email is required"}
    ref = practice_firm_ref(firm_ref)
    if ref is None:
        return {"error": "firm_ref is required (letters/digits)"}

    user = User.query.filter(db.func.lower(User.email) == email).first()
    created_user = False
    if user is None:
        user = User(username=email[:64], email=email, subscription_status="active")
        user.set_password(secrets.token_urlsafe(32))  # they reset to their own
        db.session.add(user)
        db.session.flush()
        created_user = True
    elif user.is_deleted:
        return {"error": "account is deleted"}

    link = PracticeLink.query.filter_by(accountant_user_id=user.id).first()
    if link is None:
        link = PracticeLink(accountant_user_id=user.id, firm_ref=ref,
                            firm_name=(firm_name or "").strip() or None)
        db.session.add(link)
    else:
        link.firm_ref = ref
        if firm_name:
            link.firm_name = firm_name.strip()
    db.session.commit()
    logger.info("practice link: user %s bound to firm %s (created_user=%s)",
                user.id, ref, created_user)
    return {"linked": True, "accountant_user_id": user.id,
            "firm_ref": ref, "created_user": created_user}


@provisioning.route("/api/provisioning/analee/practice", methods=["POST"])
def practice_link_ensure():
    """S2S: THE ACCOUNTANTS binds an accountant's one Analee login to their
    firm, so /practice lists all the firm's client workspaces. Same dark flag
    + fail-closed bearer as every other seam endpoint."""
    if not enabled():
        return jsonify({"error": "not found"}), 404
    ok = _bearer_ok(request.headers.get("Authorization", ""))
    if ok is None:
        return jsonify({"error": "provisioning not configured"}), 503
    if not ok:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    result = ensure_practice_link(
        data.get("accountant_email"), data.get("firm_ref"),
        data.get("firm_name"))
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result), 200


@provisioning.route("/workspace/enter", methods=["GET"])
def workspace_enter():
    """Browser lands here from THE ACCOUNTANTS; verify and log into the workspace."""
    if not enabled():
        return jsonify({"error": "not found"}), 404
    if not _secret():
        # Unconfigured → behave as if dark (nothing can have been minted).
        return jsonify({"error": "not found"}), 404
    from flask_login import login_user
    from itsdangerous import BadSignature, SignatureExpired
    from models import User

    try:
        payload = _login_serializer().loads(
            request.args.get("token", ""), max_age=_link_ttl())
    except (SignatureExpired, BadSignature):
        flash("That secure link has expired — please reopen this client "
              "from The Accountants.", "info")
        return redirect(url_for("auth.login"))
    user = User.query.get(payload.get("uid")) if isinstance(payload, dict) else None
    if (user is None or user.is_deleted
            or not _is_workspace_email(user.email)
            or user.subscription_status != "active"):
        flash("That workspace is not available — please reopen it "
              "from The Accountants.", "info")
        return redirect(url_for("auth.login"))
    login_user(user)
    session["workspace_session"] = True
    session["workspace_email"] = user.email
    return redirect(url_for("main.dashboard"))
