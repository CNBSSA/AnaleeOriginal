"""One-Login Practice Layer (P1) — "a page on top of what is already there".

Festus's direction (2026-07-18): the accountant logs into Analee ONCE and sees
My Clients — every client workspace their practice owns — and switches between
them with a click. Never fifty passwords, never in-and-out.

How it works (no frozen capability touched — the engine is only ever *called*
by the pages the accountant lands on, exactly as today):
- ``PracticeLink`` (additive table) binds the accountant's own account to their
  firm ref from THE ACCOUNTANTS (e.g. ``acc-7``).
- The firm's client workspaces are the hidden alias accounts the 07-15 seam
  already creates (``client+acc-7-42@ws.theaccountants.local``); membership is
  derived from the ref prefix — no mapping table for clients needed.
- "Open" performs the same server-side identity switch the signed
  ``/workspace/enter`` link performs, minus the S2S round-trip; the
  accountant's own user id is kept in the session so a header chip can switch
  straight back to My Clients.

Ships DARK behind ``ANALEE_PRACTICE_LAYER_ENABLED`` (default off): every route
404s while off, the chip never renders (its session keys can only be set by
these routes). Fail-soft registration in app.py — a failure here can never
stop Analee booting.
"""
import logging
import os

from flask import (Blueprint, flash, jsonify, redirect, render_template,
                   request, session, url_for)
from flask_login import current_user, login_required, login_user

logger = logging.getLogger(__name__)

practice = Blueprint("practice", __name__)

_RETURN_KEY = "practice_return_uid"


def enabled():
    return os.environ.get("ANALEE_PRACTICE_LAYER_ENABLED", "False") == "True"


def _my_link():
    """The current user's PracticeLink, or None."""
    from models import PracticeLink
    if not current_user.is_authenticated:
        return None
    return PracticeLink.query.filter_by(
        accountant_user_id=current_user.id).first()


def _firm_workspaces(firm_ref):
    """All active client workspaces belonging to ``firm_ref``.

    Prefix match includes the trailing separator so firm ``acc-1`` can never
    see firm ``acc-10``'s clients. Escapes LIKE wildcards defensively (refs
    are already sanitised to [a-z0-9-], so none should occur)."""
    from models import db, User, CompanySettings
    from provisioning import WORKSPACE_EMAIL_DOMAIN

    prefix = f"client+{firm_ref}-"
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = f"{escaped}%@{WORKSPACE_EMAIL_DOMAIN}"
    rows = (db.session.query(User, CompanySettings)
            .outerjoin(CompanySettings, CompanySettings.user_id == User.id)
            .filter(db.func.lower(User.email).like(pattern, escape="\\"),
                    User.is_deleted.is_(False),
                    User.subscription_status == "active")
            .order_by(CompanySettings.company_name)
            .all())
    domain_suffix = "@" + WORKSPACE_EMAIL_DOMAIN
    out = []
    for user, settings in rows:
        local = user.email[:-len(domain_suffix)]
        client_ref = local[len("client+"):]
        out.append({
            "workspace_user_id": user.id,
            "client_ref": client_ref,
            "company_name": (settings.company_name if settings else None)
                            or client_ref,
        })
    return out


def _workspace_belongs_to_firm(workspace_user, firm_ref):
    from provisioning import WORKSPACE_EMAIL_DOMAIN
    email = (workspace_user.email or "").lower()
    return (email.startswith(f"client+{firm_ref}-")
            and email.endswith("@" + WORKSPACE_EMAIL_DOMAIN))


@practice.route("/practice", methods=["GET"])
@login_required
def my_clients():
    """The one page: every client of the practice, one click each."""
    if not enabled():
        return jsonify({"error": "not found"}), 404
    link = _my_link()
    if link is None:
        # Not a practice accountant — the page does not exist for them.
        return jsonify({"error": "not found"}), 404
    clients = _firm_workspaces(link.firm_ref)
    return render_template("practice/my_clients.html",
                           link=link, clients=clients)


@practice.route("/practice/open/<int:workspace_user_id>", methods=["POST"])
@login_required
def open_client(workspace_user_id):
    """Switch this session into a client workspace — no password, no logout.

    Same effect as the signed /workspace/enter entry, but initiated by the
    practice's own authenticated accountant, so the only checks needed are
    ownership (ref prefix) and workspace liveness."""
    if not enabled():
        return jsonify({"error": "not found"}), 404
    link = _my_link()
    if link is None:
        return jsonify({"error": "not found"}), 404
    from models import User
    ws = User.query.get(workspace_user_id)
    if (ws is None or ws.is_deleted or ws.subscription_status != "active"
            or not _workspace_belongs_to_firm(ws, link.firm_ref)):
        flash("That client workspace is not available.", "error")
        return redirect(url_for("practice.my_clients"))

    accountant_id = current_user.id
    login_user(ws)
    session["workspace_session"] = True
    session["workspace_email"] = ws.email
    session[_RETURN_KEY] = accountant_id
    return redirect(url_for("main.dashboard"))


@practice.route("/practice/return", methods=["POST"])
def return_to_practice():
    """The chip's target: switch back from a client workspace to My Clients."""
    if not enabled():
        return jsonify({"error": "not found"}), 404
    from models import db, User, PracticeLink

    accountant_id = session.get(_RETURN_KEY)
    accountant = User.query.get(accountant_id) if accountant_id else None
    link = (PracticeLink.query.filter_by(accountant_user_id=accountant.id).first()
            if accountant is not None else None)
    if (accountant is None or accountant.is_deleted or link is None):
        session.pop(_RETURN_KEY, None)
        flash("Please sign in again to see your clients.", "info")
        return redirect(url_for("auth.login"))
    login_user(accountant)
    session.pop("workspace_session", None)
    session.pop("workspace_email", None)
    session.pop(_RETURN_KEY, None)
    return redirect(url_for("practice.my_clients"))


def register(app):
    """Fail-soft registration (club_sso pattern): a fault here logs and moves
    on — the practice layer can never stop Analee booting."""
    try:
        app.register_blueprint(practice)
        logger.info("practice layer registered (enabled=%s)", enabled())
    except Exception:  # noqa: BLE001
        logger.exception("practice layer failed to register (non-fatal)")
