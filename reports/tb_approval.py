"""Admin approval before a standalone trial balance leaves Analee.

Festus, 2026-09-28: *"Approve before a standalone trial balance goes to THE
ACCOUNTANTS."* Every transmission path — the authenticated JSON API, the
24-hour share link (minting AND fetching) and the one-click Send TB — refuses
unless an Analee administrator has approved THIS trial balance for THIS
period. The Excel download is a manual export and is not gated.

An approval is pinned to a fingerprint of the rows it approved. If a line is
recategorised or a statement imported after approval, the fingerprint no
longer matches and the trial balance must be approved again — an approval can
never silently cover numbers nobody looked at.

The frozen ``trial_balance_service`` is only CALLED here, never changed.
Switch: ``ANALEE_TB_APPROVAL_REQUIRED`` (default on; ``0`` disables the gate).
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime

from models import TrialBalanceApproval, db

STATUS_REQUESTED = 'requested'
STATUS_APPROVED = 'approved'
STATUS_DECLINED = 'declined'

NOT_APPROVED_MESSAGE = (
    'This trial balance has not been approved for sending. Request approval on '
    'the Trial Balance page; an Analee administrator approves it, and then it '
    'can go to THE ACCOUNTANTS.')
STALE_MESSAGE = (
    'This trial balance has changed since it was approved. Request approval '
    'again so an administrator can look at the current figures.')


class ApprovalRequired(ValueError):
    """Raised by ``require_approval`` when the trial balance may not be sent."""


def enabled() -> bool:
    return os.environ.get('ANALEE_TB_APPROVAL_REQUIRED', '1').strip().lower() not in (
        '0', 'false', 'no', 'off')


def fingerprint(ctx) -> str:
    """A stable digest of what the trial balance says for its period."""
    h = hashlib.sha256()
    h.update(ctx.end_date.strftime('%Y-%m-%d').encode())
    for row in sorted(ctx.rows, key=lambda r: r.link):
        h.update(f'|{row.link}={row.amount:.2f}'.encode())
    return h.hexdigest()


def latest(user_id: int, period_end) -> TrialBalanceApproval | None:
    return (TrialBalanceApproval.query
            .filter_by(user_id=user_id, period_end=_as_date(period_end))
            .order_by(TrialBalanceApproval.id.desc())
            .first())


def status_for(user_id: int, ctx) -> dict:
    """What the Trial Balance page shows: state + the record behind it."""
    record = latest(user_id, ctx.end_date)
    if record is None:
        state = 'none'
    elif record.status == STATUS_APPROVED:
        state = 'approved' if record.tb_fingerprint == fingerprint(ctx) else 'stale'
    elif record.status == STATUS_REQUESTED:
        state = 'requested' if record.tb_fingerprint == fingerprint(ctx) else 'stale'
    else:
        state = 'declined'
    return {'state': state, 'record': record, 'required': enabled()}


def is_approved(user_id: int, ctx) -> bool:
    record = latest(user_id, ctx.end_date)
    return (record is not None and record.status == STATUS_APPROVED
            and record.tb_fingerprint == fingerprint(ctx))


def require_approval(user_id: int, ctx) -> None:
    """Raise ``ApprovalRequired`` unless this exact trial balance is approved."""
    if not enabled():
        return
    record = latest(user_id, ctx.end_date)
    if record is not None and record.status == STATUS_APPROVED:
        if record.tb_fingerprint == fingerprint(ctx):
            return
        raise ApprovalRequired(STALE_MESSAGE)
    raise ApprovalRequired(NOT_APPROVED_MESSAGE)


def request_approval(user_id: int, ctx, *, requested_by: int) -> TrialBalanceApproval:
    """Record a request for the current figures (idempotent for identical ones)."""
    record = latest(user_id, ctx.end_date)
    fp = fingerprint(ctx)
    if record is not None and record.tb_fingerprint == fp and record.status in (
            STATUS_REQUESTED, STATUS_APPROVED):
        return record
    record = TrialBalanceApproval(
        user_id=user_id,
        period_end=_as_date(ctx.end_date),
        period_start=_as_date(ctx.start_date),
        tb_fingerprint=fp,
        row_count=len(ctx.rows),
        total_debits=float(ctx.total_debits),
        total_credits=float(ctx.total_credits),
        status=STATUS_REQUESTED,
        requested_by=requested_by,
        requested_at=datetime.utcnow(),
    )
    db.session.add(record)
    db.session.commit()
    return record


def decide(record: TrialBalanceApproval, *, admin_id: int, approve: bool,
           note: str = '') -> TrialBalanceApproval:
    record.status = STATUS_APPROVED if approve else STATUS_DECLINED
    record.decided_by = admin_id
    record.decided_at = datetime.utcnow()
    record.note = (note or '')[:500]
    db.session.commit()
    return record


def _as_date(value):
    return value.date() if isinstance(value, datetime) else value
