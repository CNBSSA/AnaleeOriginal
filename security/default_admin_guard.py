"""Neutralise the legacy default admin credential if it is still live.

Background (docs/QA_AUDIT_2026-09-04.md, item #1): an old code path created an
admin user ``festusa@cnbs.co.za`` with the password ``admin123``. The code was
deleted (``71eef5a``), but deleting code does not unwrite a database row. Only
the live database knows whether that row still carries the default password.

This guard answers that without a login attempt and without exposing anything:
it loads the user, tests the stored hash against the known default, and ONLY if
it matches sets a new password. A row with any other password is never touched.
Idempotent: once rotated, every later run is a no-op.

The replacement password is ``ADMIN_PASSWORD`` from the environment when set
(the same variable ``create_admin.py`` uses, so the operator can choose it in
advance), otherwise a random one nobody knows. Recovery after a random rotation
needs no shell: set ``ADMIN_PASSWORD`` plus the one-shot flag
``ADMIN_PASSWORD_FORCE_RESET=1``, redeploy, then remove the flag — with the flag
the guard applies ``ADMIN_PASSWORD`` to the legacy admin row regardless of its
current password (``force=True``).

No password, old or new, is ever logged or printed.
"""
from __future__ import annotations

import logging
import secrets

from models import User

LEGACY_ADMIN_EMAIL = 'festusa@cnbs.co.za'
LEGACY_ADMIN_PASSWORD = 'admin123'

STATUS_ABSENT = 'absent'            # no such user row
STATUS_NOT_DEFAULT = 'not_default'  # row exists, password is not the default
STATUS_ROTATED = 'rotated'          # default password was live; now replaced
STATUS_FORCED = 'forced'            # operator-requested reset to ADMIN_PASSWORD

logger = logging.getLogger(__name__)


def neutralise_default_admin(session, *, replacement_password: str | None = None,
                             email: str = LEGACY_ADMIN_EMAIL, force: bool = False) -> dict:
    """Rotate the legacy admin's password iff it is still the known default.

    With ``force=True`` AND a ``replacement_password``, apply that password to
    the row regardless of its current value (the operator's deliberate recovery
    step). ``force`` without a replacement password does nothing extra — the
    guard will never force a random password onto a working account.

    Returns ``{'status': ..., 'email': ..., 'used_env_password': bool}``.
    Never raises on a clean database state; database errors propagate so the
    caller can report them.
    """
    user = session.query(User).filter_by(email=email).first()
    if user is None:
        return {'status': STATUS_ABSENT, 'email': email, 'used_env_password': False}

    if force and replacement_password:
        user.set_password(replacement_password)
        session.commit()
        logger.warning("admin password for %s reset to ADMIN_PASSWORD on operator request "
                       "(ADMIN_PASSWORD_FORCE_RESET) — remove the flag now", email)
        return {'status': STATUS_FORCED, 'email': email, 'used_env_password': True}

    if not user.check_password(LEGACY_ADMIN_PASSWORD):
        return {'status': STATUS_NOT_DEFAULT, 'email': email, 'used_env_password': False}

    used_env = bool(replacement_password)
    new_password = replacement_password or secrets.token_urlsafe(32)
    user.set_password(new_password)
    session.commit()
    logger.warning(
        "default admin credential was LIVE for %s and has been rotated (%s)",
        email, 'to ADMIN_PASSWORD from the environment' if used_env else 'to a random value')
    return {'status': STATUS_ROTATED, 'email': email, 'used_env_password': used_env}
