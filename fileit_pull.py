"""FileIt → Analee bank-statement tunnel (Festus 2026-09-28, authorised
exception: "The documents must have tunnels to all other necessary products";
tunnel 2 of the plan).

A client's bank statements already filed in FileIt are pulled into that
client's Analee workspace instead of being uploaded a second time:

* CSV / xlsx → the SAME ``BankStatementService.process_upload`` the upload
  page uses (SA format detection, bank stamping, auto-process untouched);
* PDF → the SAME extraction + review screen the PDF page uses, so the
  unchanged ``/ocr/statement/confirm`` finishes the import.

Whose folder: only a WORKSPACE session knows its client — the alias user
``client+<ref>@ws.theaccountants.local``. The ref is asked of the Practice Club
roster (``/practice/resolve/`` with product ``analee``), which answers with the
sibling FileIt folder id. Ordinary subscriber sessions get nothing new.

DARK behind ``ANALEE_FILEIT_PULL_ENABLED``. Nothing frozen is touched: the
engine is reached only through the import paths that already call it.
"""
from __future__ import annotations

import io
import logging
import os

import requests
from flask import (Blueprint, abort, current_app, flash, redirect,
                   render_template, request, session, url_for)
from flask_login import current_user, login_required
from werkzeug.datastructures import FileStorage

logger = logging.getLogger(__name__)

fileit = Blueprint("fileit", __name__, url_prefix="/fileit")

_TIMEOUT_SECONDS = 30
_HUB_TIMEOUT_SECONDS = 10
_API = "/api/integration/v1"
_SOURCE_PRODUCT = "analee"
_TARGET_PRODUCT = "bookstorage"
TABLE_EXTENSIONS = (".csv", ".xlsx")
PDF_EXTENSIONS = (".pdf",)


class FileItError(Exception):
    """A plain-language reason the tunnel could not deliver."""


def enabled() -> bool:
    return os.environ.get("ANALEE_FILEIT_PULL_ENABLED", "False") == "True"


def configured() -> bool:
    # The credential normally arrives from the hub per practice (in the resolve
    # answer); a service-wide client id + secret or a static token are fallbacks.
    return bool(os.environ.get("FILEIT_API_BASE_URL", "").strip()
                and os.environ.get("CLUB_PRACTICE_SYNC_URL", "").strip()
                and os.environ.get("CLUB_PRACTICE_SYNC_TOKEN", "").strip())


def _base() -> str:
    return os.environ.get("FILEIT_API_BASE_URL", "").strip().rstrip("/")


# --- credentials: the practice's own key, never another practice's -------------
# Festus 2026-09-30: access is provisioned in FileIt; one user's documents must not
# leak into another user's applications. The hub hands us, in the resolve answer,
# the FileIt credential issued for THIS practice and Analee; kept for the current
# request only (thread-local), exchanged for a short-lived token cached per client
# id. Fallbacks: FILEIT_CLIENT_ID + FILEIT_CLIENT_SECRET, then FILEIT_API_TOKEN.
import threading as _threading
import time as _time

_ctx = _threading.local()
_tokens_by_client: dict = {}


def _remember_credential(data: dict) -> None:
    cred = data.get("credential") if isinstance(data, dict) else None
    ok = isinstance(cred, dict) and cred.get("client_id") and cred.get("client_secret")
    _ctx.credential = dict(cred) if ok else None


def _token_for(client_id: str, secret: str, base: str) -> str:
    hit = _tokens_by_client.get(client_id)
    if hit and _time.time() < hit[1]:
        return hit[0]
    resp = requests.post(f"{base}{_API}/auth/token",
                         json={"client_id": client_id, "client_secret": secret},
                         timeout=_TIMEOUT_SECONDS, allow_redirects=False)
    resp.raise_for_status()
    data = resp.json() or {}
    token = str(data.get("access_token") or "")
    ttl = int(data.get("expires_in") or 3600)
    _tokens_by_client[client_id] = (token, _time.time() + max(60, ttl - 120))
    return token


def _bearer() -> str:
    cred = getattr(_ctx, "credential", None)
    if cred:
        return _token_for(cred["client_id"], cred["client_secret"],
                          (cred.get("base_url") or _base()).rstrip("/"))
    client_id = os.environ.get("FILEIT_CLIENT_ID", "").strip()
    secret = os.environ.get("FILEIT_CLIENT_SECRET", "").strip()
    if client_id and secret:
        return _token_for(client_id, secret, _base())
    return os.environ.get("FILEIT_API_TOKEN", "").strip()


def _headers() -> dict:
    return {"Authorization": f"Bearer {_bearer()}", "User-Agent": "Analee/fileit-pull"}


def in_workspace() -> bool:
    from provisioning import _is_workspace_email
    return bool(session.get("workspace_session")
                and _is_workspace_email(getattr(current_user, "email", "") or ""))


def workspace_ref() -> str:
    from provisioning import WORKSPACE_EMAIL_DOMAIN
    local = (current_user.email or "")[:-(len(WORKSPACE_EMAIL_DOMAIN) + 1)]
    return local[len("client+"):] if local.startswith("client+") else ""


def available() -> bool:
    """The button shows only when it can do something: flag on, in a workspace."""
    try:
        return enabled() and current_user.is_authenticated and in_workspace()
    except Exception:  # noqa: BLE001 — a template helper must never raise
        return False


# --- the hub roster: which FileIt folder is this client's ------------------

def _hub_endpoint(name: str) -> str:
    url = os.environ.get("CLUB_PRACTICE_SYNC_URL", "").strip()
    if not url:
        return ""
    base = url.rstrip("/")
    if base.endswith("/sync"):
        base = base[: -len("/sync")]
    return f"{base}/{name}/"


def resolve_folder(ref: str) -> str:
    _ctx.credential = None  # never carry one practice's key into another's request
    endpoint = _hub_endpoint("resolve")
    token = os.environ.get("CLUB_PRACTICE_SYNC_TOKEN", "").strip()
    if not endpoint or not token or not ref:
        return ""
    try:
        resp = requests.post(
            endpoint,
            json={"product": _SOURCE_PRODUCT, "external_ref": ref,
                  "target_product": _TARGET_PRODUCT},
            headers={"Authorization": f"Bearer {token}"},
            timeout=_HUB_TIMEOUT_SECONDS, allow_redirects=False)
    except requests.RequestException as exc:
        logger.warning("FileIt folder resolve failed for %s: %s", ref, exc)
        return ""
    if resp.status_code != 200:
        return ""
    try:
        data = resp.json() or {}
    except ValueError:
        return ""
    _remember_credential(data)
    return str(data.get("external_ref") or "") if data.get("found") else ""


# --- FileIt's integration API ---------------------------------------------

def _get(path: str, **kwargs) -> requests.Response:
    resp = requests.get(f"{_base()}{_API}{path}", headers=_headers(),
                        timeout=_TIMEOUT_SECONDS, allow_redirects=False, **kwargs)
    resp.raise_for_status()
    return resp


def list_folder_documents(folder_id: str) -> list[dict]:
    data = _get(f"/folders/{folder_id}/documents", params={"limit": 500}).json() or {}
    # FileIt's list lives under ``items``; ``documents`` kept as a fallback.
    if isinstance(data, dict):
        docs = data.get("items")
        if docs is None:
            docs = data.get("documents")
    else:
        docs = data
    return list(docs or [])


def route_for(doc: dict) -> str | None:
    name = (doc.get("original_filename") or doc.get("filename") or "").lower()
    if name.endswith(TABLE_EXTENSIONS):
        return "table"
    if name.endswith(PDF_EXTENSIONS) or (doc.get("category") or "") == "pdf":
        return "pdf"
    return None


def usable_documents() -> list[dict]:
    """Bank statements Analee can import from this workspace's FileIt folder,
    FileIt-classified statements first, then newest first."""
    if not configured():
        raise FileItError("FileIt is not connected on this service yet "
                          "(FILEIT_API_BASE_URL / FILEIT_API_TOKEN).")
    ref = workspace_ref()
    folder_id = resolve_folder(ref)
    if not folder_id:
        raise FileItError("This client is not linked to a FileIt folder on the "
                          "Practice Club roster. Link the client to FileIt in My "
                          "Practice first.")
    try:
        docs = list_folder_documents(folder_id)
    except requests.RequestException as exc:
        logger.warning("FileIt listing failed for %s: %s", ref, exc)
        raise FileItError("FileIt could not be reached, or this service's FileIt "
                          "token has no access to the client's folder. Check the "
                          "grant in FileIt → Admin → Integrations.") from exc
    usable = []
    for d in docs:
        route = route_for(d)
        if route:
            usable.append({**d, "route": route,
                           "display_name": d.get("original_filename")
                           or d.get("filename") or f"document {d.get('id')}",
                           "is_statement": d.get("document_kind") == "bank_statement"})
    usable.sort(key=lambda d: (not d["is_statement"], d.get("created_at") or ""),
                reverse=False)
    usable.sort(key=lambda d: d.get("created_at") or "", reverse=True)
    usable.sort(key=lambda d: not d["is_statement"])
    return usable


def fetch(document_id: str) -> tuple[dict, bytes]:
    """The listing entry and bytes — only if the document is in this
    workspace's own folder (tenancy guard)."""
    docs = usable_documents()
    match = next((d for d in docs if str(d.get("id")) == str(document_id)), None)
    if match is None:
        raise FileItError("That document is not in this client's FileIt folder.")
    try:
        content = _get(f"/documents/{document_id}/content").content
    except requests.RequestException as exc:
        logger.warning("FileIt fetch failed for doc %s: %s", document_id, exc)
        raise FileItError("FileIt could not deliver that document right now.") from exc
    return match, content


# --- routes ------------------------------------------------------------------

def _bank_accounts():
    from models import Account
    return (Account.query.filter(Account.user_id == current_user.id,
                                 Account.link.like("ca.810%"))
            .order_by(Account.name).all())


@fileit.route("/statements", methods=["GET"])
@login_required
def statements():
    if not enabled():
        abort(404)
    if not in_workspace():
        flash("Open a client first, then pull their bank statements from FileIt.", "info")
        return redirect(url_for("main.dashboard"))
    error, documents = "", []
    try:
        documents = usable_documents()
    except FileItError as exc:
        error = str(exc)
    return render_template("fileit/statements.html", documents=documents,
                           error=error, accounts=_bank_accounts())


@fileit.route("/statements/import", methods=["POST"])
@login_required
def import_statement():
    if not enabled():
        abort(404)
    if not in_workspace():
        flash("Open a client first, then pull their bank statements from FileIt.", "info")
        return redirect(url_for("main.dashboard"))
    document_id = (request.form.get("document_id") or "").strip()
    account_id = (request.form.get("account_id") or "").strip()
    accounts = _bank_accounts()
    if not any(str(a.id) == account_id for a in accounts):
        flash("Choose which bank account this statement belongs to.", "error")
        return redirect(url_for("fileit.statements"))
    try:
        doc, content = fetch(document_id)
    except FileItError as exc:
        flash(str(exc), "error")
        return redirect(url_for("fileit.statements"))
    name = doc["display_name"]

    if doc["route"] == "table":
        from bank_statements.services import BankStatementService
        storage = FileStorage(stream=io.BytesIO(content), filename=name)
        success, response = BankStatementService().process_upload(
            file=storage, account_id=int(account_id), user_id=current_user.id)
        if not success:
            flash(response.get("error", "Import failed."), "error")
            for detail in response.get("details") or []:
                flash(detail, "warning")
            return redirect(url_for("fileit.statements"))
        started = False
        if response.get("file_id"):
            try:
                from services.auto_process import schedule_file_autoprocess
                started = schedule_file_autoprocess(
                    current_app._get_current_object(), response["file_id"], current_user.id)
            except Exception:  # noqa: BLE001
                logger.exception("Could not schedule auto-process")
        flash(f"Imported {name} from FileIt.", "success")
        if started:
            flash("Analee is categorising and explaining the rows now — open "
                  "Analyze Data in a minute to review what needs your eye.", "info")
        return redirect(url_for("bank_statements.upload"))

    # PDF: the same extraction + review screen as the PDF upload page; the
    # unchanged confirm step finishes the import.
    from ocr.routes import review_extracted_statement
    return review_extracted_statement(content, name, account_id,
                                      request.form.get("opening_balance"),
                                      request.form.get("closing_balance"),
                                      back_to="fileit.statements")


def register(app):
    """Fail-soft registration (practice_layer pattern): the blueprint is always
    registered so the manifest is stable; routes 404 while the flag is off."""
    try:
        app.register_blueprint(fileit)
        app.context_processor(lambda: {"fileit_pull_available": available()})
        logger.info("FileIt pull registered (enabled=%s)", enabled())
    except Exception:  # noqa: BLE001
        logger.exception("FileIt pull failed to register (non-fatal)")
