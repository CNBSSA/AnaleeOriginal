"""Analee → FileIt trial-balance tunnel (Festus 2026-09-29, authorised
exception: "All South African products that are in the clubhouse and clubhub
should be included"). The SEND direction beside ``fileit_pull`` (TAKE).

One button on the Trial Balance page, inside an accountant's client
workspace: file this period's trial balance Excel into the client's FileIt
folder.

* The workbook is the SAME one "Download Excel" gives —
  ``build_booksxperts_trial_balance_xlsx`` on ``load_trial_balance`` — CALLED,
  never re-computed. The frozen TB core is untouched.
* Filing a trial balance into FileIt is a transmission out of Analee, so the
  SAME admin approval gate as the share link and Send TB applies
  (``reports.tb_approval.require_approval``): refused with the same plain
  message until an administrator has approved these exact figures.
* Whose folder: only a WORKSPACE session, resolved through the Practice Club
  roster with the workspace ref — the same functions ``fileit_pull`` uses
  (credential handling included; nothing is duplicated here).
* Idempotent: the Idempotency-Key is derived from (source, record, tenant,
  approved-figures fingerprint), so the same trial balance is never filed
  twice even though a freshly built workbook's bytes differ by timestamp.

DARK behind ``ANALEE_FILEIT_PUSH_ENABLED`` (default off): the route 404s and
the button does not render.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os

import requests
from flask import Blueprint, abort, flash, redirect, request, url_for
from flask_login import current_user, login_required

import fileit_pull

logger = logging.getLogger(__name__)

fileit_push = Blueprint("fileit_push", __name__, url_prefix="/fileit")

DOCUMENT_KIND = "general"  # FileIt has no trial-balance kind; closest real one.
_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def enabled() -> bool:
    return os.environ.get("ANALEE_FILEIT_PUSH_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on")


def available() -> bool:
    """The button shows only when it can do something: flag on, in a workspace."""
    try:
        return enabled() and current_user.is_authenticated and fileit_pull.in_workspace()
    except Exception:  # noqa: BLE001 — a template helper must never raise
        return False


def idempotency_key(ref: str, user_id: int, ctx) -> str:
    """Deterministic per (source, record, tenant, figures). ≤ 128 chars."""
    from reports import tb_approval
    raw = "|".join(["analee", "trial_balance", ref, str(user_id),
                    ctx.end_date.strftime("%Y-%m-%d"), tb_approval.fingerprint(ctx)])
    return "analee-tb-" + hashlib.sha256(raw.encode()).hexdigest()


def _audit(outcome: str, ref: str, detail: str = "") -> None:
    # Analee has no audit-log table; every attempt is written to the log as a
    # single structured line (filed, already there, or refused and why).
    logger.info("AUDIT fileit.trial_balance.push outcome=%s user=%s ref=%s %s",
                outcome, getattr(current_user, "id", None), ref, detail)


class _Refused(Exception):
    """A plain-language reason the trial balance was not filed."""


def _refusal_for(status: int) -> str:
    if status == 401:
        return ("FileIt did not accept this service's credential. Check the "
                "integration client in FileIt → Admin → Integrations.")
    if status == 403:
        return ("FileIt refused: this service may not write to the client's "
                "folder (no write grant, or the client is closed in FileIt).")
    if status == 404:
        return "FileIt could not find the client's folder. Re-link the client in My Practice."
    if status == 413:
        return "FileIt refused the file as too large."
    return f"FileIt could not file the trial balance right now (status {status})."


def file_trial_balance(ref: str, *, filename: str, content: bytes, ctx, key: str) -> str:
    """Send the workbook to this client's folder. Returns 'filed' or 'already'.
    Raises ``_Refused`` with a plain message; never anything else."""
    if not fileit_pull.configured():
        raise _Refused("FileIt is not connected on this service yet "
                       "(FILEIT_API_BASE_URL and CLUB_PRACTICE_SYNC_URL / CLUB_PRACTICE_SYNC_TOKEN).")
    base = fileit_pull._base()
    if not base.startswith("https://"):
        raise _Refused("FileIt must be reached over HTTPS; FILEIT_API_BASE_URL is not an https:// address.")
    folder_id = fileit_pull.resolve_folder(ref)
    if not folder_id:
        raise _Refused("This client is not linked to a FileIt folder on the "
                       "Practice Club roster. Link the client to FileIt in My "
                       "Practice first.")
    metadata = {
        "type": "trial_balance",
        "period_start": ctx.start_date.strftime("%Y-%m-%d"),
        "period_end": ctx.end_date.strftime("%Y-%m-%d"),
        "row_count": len(ctx.rows),
    }
    try:
        headers = dict(fileit_pull._headers(), **{"Idempotency-Key": key})
        headers["User-Agent"] = "Analee/fileit-push"
        resp = requests.post(
            f"{base}{fileit_pull._API}/folders/{folder_id}/documents",
            headers=headers,
            files={"file": (filename, content, _XLSX)},
            data={"document_kind": DOCUMENT_KIND, "source_app": "analee",
                  "metadata": json.dumps(metadata), "external_reference": key},
            timeout=fileit_pull._TIMEOUT_SECONDS, allow_redirects=False)
    except requests.RequestException as exc:
        logger.warning("FileIt trial-balance push failed for %s: %s", ref, exc)
        raise _Refused("FileIt could not be reached right now. Try again in a minute.") from exc
    if resp.status_code == 201:
        return "filed"
    if resp.status_code == 200:
        return "already"
    raise _Refused(_refusal_for(resp.status_code))


@fileit_push.route("/trial-balance", methods=["POST"])
@login_required
def send_trial_balance():
    if not enabled():
        abort(404)
    if not fileit_pull.in_workspace():
        flash("Open a client first, then file their trial balance in FileIt.", "info")
        return redirect(url_for("main.dashboard"))

    from models import CompanySettings
    from reports import tb_approval
    from reports.routes import BadPeriodError, _period_query_string, _requested_period
    from reports.trial_balance_service import (build_booksxperts_trial_balance_xlsx,
                                               export_filename, load_trial_balance)
    back = url_for("reports.trial_balance", **_period_query_string())
    ref = fileit_pull.workspace_ref()
    try:
        settings = CompanySettings.query.filter_by(user_id=current_user.id).first()
        if settings is None:
            raise _Refused("Please configure company settings first.")
        ctx = load_trial_balance(current_user.id, **_requested_period(current_user.id))
        if not ctx.rows:
            raise _Refused("There is no trial balance to file for this period yet.")
        try:
            tb_approval.require_approval(current_user.id, ctx)
        except tb_approval.ApprovalRequired as exc:
            raise _Refused(str(exc)) from exc
        content = build_booksxperts_trial_balance_xlsx(
            ctx.rows, company_name=settings.company_name, period_end=ctx.end_date)
        filename = export_filename(ctx.end_date)
        key = idempotency_key(ref, current_user.id, ctx)
        outcome = file_trial_balance(ref, filename=filename, content=content,
                                     ctx=ctx, key=key)
    except (_Refused, BadPeriodError) as exc:
        _audit("refused", ref, f"reason={exc}")
        flash(str(exc), "error")
        return redirect(back)
    except Exception:  # noqa: BLE001 — never a 500
        logger.exception("FileIt trial-balance push failed")
        _audit("refused", ref, "reason=unexpected error")
        flash("The trial balance could not be filed in FileIt right now.", "error")
        return redirect(back)

    _audit(outcome, ref, f"file={filename} key={key}")
    if outcome == "already":
        flash(f"{filename} is already in the client's FileIt folder — not filed twice.", "info")
    else:
        flash(f"Filed {filename} in the client's FileIt folder.", "success")
    return redirect(back)


def register(app):
    """Fail-soft registration: always registered so the manifest is stable;
    the route 404s while the flag is off."""
    try:
        app.register_blueprint(fileit_push)
        app.context_processor(lambda: {"fileit_push_available": available()})
        logger.info("FileIt push registered (enabled=%s)", enabled())
    except Exception:  # noqa: BLE001
        logger.exception("FileIt push failed to register (non-fatal)")
