"""Login abuse protection: a database-backed failed-attempt ledger (work-orders #69, R1).

Password login had no throttle at all. With two-factor authentication ruled out
for every product, this is the compensating control that must exist.

How it works
- Before a password is checked, the login route asks ``locked(email)``: too many
  FAILED attempts in the last WINDOW for that e-mail (EMAIL_LIMIT) or from that
  client address (IP_LIMIT) → the attempt is refused with a plain message and
  HTTP 429, whatever the password.
- Every failure is recorded; a success clears that e-mail's failures (the IP's
  stay, so a password spray across many addresses still runs into IP_LIMIT).
- Keys are SHA-256 hashes of the normalised address / IP — the ledger never
  stores an e-mail or an IP address in clear.
- The ledger is a TABLE, not an in-process counter, because the deploy runs
  several gunicorn workers: a per-worker counter would multiply the limit by the
  worker count.
- FAIL-OPEN: if the table cannot be read or written, login proceeds exactly as
  it did before this existed and the error is logged. A broken ledger must never
  lock every customer out.

Recovery for a locked-out person: wait WINDOW (15 minutes); rows older than the
window stop counting. No password is ever changed by this module.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta

from flask import request

from models import db, LoginAttempt

logger = logging.getLogger(__name__)

WINDOW = timedelta(minutes=15)
EMAIL_LIMIT = 10   # failed attempts per e-mail address inside WINDOW
IP_LIMIT = 30      # failed attempts per client address inside WINDOW

LOCKED_MESSAGE = (
    'Too many sign-in attempts. Please wait 15 minutes and try again. If you '
    'cannot remember your password, use the Forgot Password link below.'
)

KIND_EMAIL = 'email'
KIND_IP = 'ip'


def client_ip() -> str:
    """The real client address. Railway terminates TLS in front of gunicorn and
    this app has no ProxyFix, so ``remote_addr`` is the proxy; the edge appends
    the client to X-Forwarded-For, so its LAST entry is the client as the edge
    saw it (a value the client itself cannot choose)."""
    forwarded = (request.headers.get('X-Forwarded-For') or '').strip()
    if forwarded:
        last = forwarded.split(',')[-1].strip()
        if last:
            return last
    return request.remote_addr or 'unknown'


def _key(kind: str, value: str) -> str:
    return hashlib.sha256(f'{kind}:{value}'.encode('utf-8')).hexdigest()


def _email_key(email: str) -> str:
    return _key(KIND_EMAIL, (email or '').strip().lower())


def _ip_key(ip: str) -> str:
    return _key(KIND_IP, (ip or '').strip())


def _count_failures(key_hash: str, since: datetime) -> int:
    return (LoginAttempt.query
            .filter(LoginAttempt.key_hash == key_hash,
                    LoginAttempt.succeeded.is_(False),
                    LoginAttempt.attempted_at >= since)
            .count())


def _insert(kind: str, key_hash: str, succeeded: bool) -> None:
    db.session.add(LoginAttempt(kind=kind, key_hash=key_hash, succeeded=succeeded,
                                attempted_at=datetime.utcnow()))
    db.session.commit()


def locked(email: str) -> bool:
    """True when this e-mail or this client address has used up its failures.
    Never raises: any database error is logged and answered False (fail-open)."""
    since = datetime.utcnow() - WINDOW
    try:
        if _count_failures(_email_key(email), since) >= EMAIL_LIMIT:
            logger.warning('login throttle: e-mail key locked (too many failures)')
            return True
        if _count_failures(_ip_key(client_ip()), since) >= IP_LIMIT:
            logger.warning('login throttle: client address locked (too many failures)')
            return True
    except Exception as exc:  # noqa: BLE001 — fail open, never lock everyone out
        logger.error('login throttle: ledger unavailable, allowing the attempt (%s)',
                     exc.__class__.__name__)
        try:
            db.session.rollback()
        except Exception:  # noqa: BLE001
            pass
    return False


def record_failure(email: str) -> None:
    """Remember a failed attempt for the e-mail AND the client address."""
    try:
        _insert(KIND_EMAIL, _email_key(email), False)
        _insert(KIND_IP, _ip_key(client_ip()), False)
        _prune()
    except Exception as exc:  # noqa: BLE001 — fail open
        logger.error('login throttle: could not record a failed attempt (%s)',
                     exc.__class__.__name__)
        try:
            db.session.rollback()
        except Exception:  # noqa: BLE001
            pass


def record_success(email: str) -> None:
    """A correct password clears that e-mail's failures. The client address's
    failures are kept so a spray over many addresses still hits IP_LIMIT."""
    try:
        (LoginAttempt.query
         .filter(LoginAttempt.key_hash == _email_key(email),
                 LoginAttempt.succeeded.is_(False))
         .delete(synchronize_session=False))
        db.session.commit()
    except Exception as exc:  # noqa: BLE001 — fail open
        logger.error('login throttle: could not clear failures after a login (%s)',
                     exc.__class__.__name__)
        try:
            db.session.rollback()
        except Exception:  # noqa: BLE001
            pass


def retry_after_seconds() -> int:
    return int(WINDOW.total_seconds())


def _prune() -> None:
    """Rows older than the window can never count again; keep the table small."""
    cutoff = datetime.utcnow() - WINDOW
    (LoginAttempt.query
     .filter(LoginAttempt.attempted_at < cutoff)
     .delete(synchronize_session=False))
    db.session.commit()
