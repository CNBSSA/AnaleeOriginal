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

import requests
from flask import (Blueprint, current_app, flash, jsonify, redirect,
                   render_template, request, session, url_for)
from flask_login import current_user, login_required, login_user

logger = logging.getLogger(__name__)

practice = Blueprint("practice", __name__)

_RETURN_KEY = "practice_return_uid"
_S2S_TIMEOUT = 10

# THE ACCOUNTANTS' entity choices (its canon — the client is created THERE).
ENTITY_CHOICES = [
    ("pty_ltd", "Private Company (Pty) Ltd"),
    ("cc", "Close Corporation"),
    ("trust", "Trust"),
    ("sole_prop", "Sole Proprietor"),
    ("partnership", "Partnership"),
    ("npo", "Non-Profit"),
]


def enabled():
    return os.environ.get("ANALEE_PRACTICE_LAYER_ENABLED", "False") == "True"


def _accountants_api():
    """Base URL of THE ACCOUNTANTS' practice API (e.g.
    ``https://theaccountants.example/api/practice``). Empty → the reverse
    door (Add client / Send TB) is unavailable, listing/switching still works."""
    return (os.environ.get("ACCOUNTANTS_PRACTICE_API_URL", "") or "").rstrip("/")


def _accountants_post(path_suffix, payload):
    """S2S call to THE ACCOUNTANTS with the shared seam bearer.

    Returns (json_dict, None) or (None, friendly_error)."""
    from provisioning import _secret

    base = _accountants_api()
    if not base:
        return None, ("This feature needs the connection to THE ACCOUNTANTS "
                      "to be configured.")
    try:
        response = requests.post(
            base + path_suffix, json=payload, timeout=_S2S_TIMEOUT,
            headers={"Authorization": f"Bearer {_secret()}",
                     "User-Agent": "Analee-PracticeLayer/1.0"})
    except requests.RequestException as exc:
        logger.warning("practice S2S %s failed: %s", path_suffix, exc)
        return None, ("Could not reach THE ACCOUNTANTS. Please try again in "
                      "a moment.")
    if response.status_code != 200:
        logger.warning("practice S2S %s returned HTTP %s",
                       path_suffix, response.status_code)
        return None, ("THE ACCOUNTANTS could not complete that right now. "
                      "Please try again.")
    try:
        return response.json(), None
    except ValueError:
        return None, "THE ACCOUNTANTS returned an unexpected response."


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
    from models import PracticeClientMeta

    rows = (db.session.query(User, CompanySettings, PracticeClientMeta)
            .outerjoin(CompanySettings, CompanySettings.user_id == User.id)
            .outerjoin(PracticeClientMeta,
                       PracticeClientMeta.workspace_user_id == User.id)
            .filter(db.func.lower(User.email).like(pattern, escape="\\"),
                    User.is_deleted.is_(False),
                    User.subscription_status == "active")
            .order_by(CompanySettings.company_name)
            .all())
    domain_suffix = "@" + WORKSPACE_EMAIL_DOMAIN
    out = []
    for user, settings, meta in rows:
        local = user.email[:-len(domain_suffix)]
        client_ref = local[len("client+"):]
        out.append({
            "workspace_user_id": user.id,
            "client_ref": client_ref,
            "company_name": (settings.company_name if settings else None)
                            or client_ref,
            "client_number": meta.client_number if meta else "",
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
                           link=link, clients=clients,
                           entity_choices=ENTITY_CHOICES,
                           add_client_available=bool(_accountants_api()))


@practice.route("/practice/add", methods=["POST"])
@login_required
def add_client():
    """P2: Add a client from inside Analee — the reverse door.

    The client is CREATED in THE ACCOUNTANTS (canonical identity: it mints
    the ref and the visible ACC-number), then the workspace is provisioned
    locally in the same request. Two doors, one truth."""
    if not enabled():
        return jsonify({"error": "not found"}), 404
    link = _my_link()
    if link is None:
        return jsonify({"error": "not found"}), 404

    name = (request.form.get("name") or "").strip()
    entity_type = (request.form.get("entity_type") or "").strip()
    if not name:
        flash("Give the client a name.", "error")
        return redirect(url_for("practice.my_clients"))
    if entity_type not in {value for value, _ in ENTITY_CHOICES}:
        entity_type = "pty_ltd"

    body, err = _accountants_post("/clients", {
        "firm_ref": link.firm_ref, "name": name, "entity_type": entity_type})
    if err:
        flash(err, "error")
        return redirect(url_for("practice.my_clients"))

    from provisioning import ensure_workspace
    result = ensure_workspace(
        body.get("client_ref"), body.get("name") or name,
        body.get("analee_entity_name"), body.get("client_number"))
    if "error" in result:
        flash("The client was registered in THE ACCOUNTANTS but the "
              "workspace could not be created — open it from the client's "
              "page there to finish setup.", "warning")
        return redirect(url_for("practice.my_clients"))

    number = body.get("client_number") or ""
    flash(f"{name}{f' ({number})' if number else ''} added — in both Analee "
          "and THE ACCOUNTANTS.", "success")
    return redirect(url_for("practice.my_clients"))


@practice.route("/practice/send-tb", methods=["POST"])
@login_required
def send_tb():
    """P3: one click inside a client workspace — send the trial balance to
    THE ACCOUNTANTS. Semi-manual by design (Festus): this hands over a signed
    share link; the workstation stages it for review-and-confirm before any
    number enters the books. The frozen TB core is only ever CALLED (the same
    share machinery as Copy-share-link)."""
    from provisioning import WORKSPACE_EMAIL_DOMAIN, _is_workspace_email

    if not enabled():
        return jsonify({"error": "not found"}), 404
    if not (session.get("workspace_session")
            and _is_workspace_email(getattr(current_user, "email", ""))):
        flash("Open a client first, then send their trial balance.", "info")
        return redirect(url_for("main.dashboard"))

    from reports.tb_share_tokens import create_share_token
    token = create_share_token(
        current_user.id, secret_key=current_app.config["SECRET_KEY"])
    share_url = url_for("reports.trial_balance_shared", token=token,
                        _external=True)

    local = current_user.email[:-(len(WORKSPACE_EMAIL_DOMAIN) + 1)]
    client_ref = local[len("client+"):]
    body, err = _accountants_post("/tb-drop", {
        "client_ref": client_ref, "share_url": share_url})
    if err:
        flash(err, "error")
        return redirect(url_for("main.dashboard"))
    number = (body or {}).get("client_number") or ""
    flash("Trial balance sent to THE ACCOUNTANTS"
          f"{f' for {number}' if number else ''} — it is waiting on the "
          "client's financial-year page for your review.", "success")
    return redirect(url_for("main.dashboard"))


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
