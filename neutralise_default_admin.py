"""Pre-deploy guard: rotate the legacy default admin password if it is still live.

Run automatically by Railway (``preDeployCommand`` in railway.json) before each
new deployment starts serving. Prints exactly one status line to the deploy
logs; never prints a password. Always exits 0 so a database hiccup here can
never block a deploy — the line in the logs is the signal.

    python neutralise_default_admin.py

Set ``ADMIN_PASSWORD`` in the environment beforehand if you want the rotation
(when one is needed) to land on a password you know. To recover the account
after a random rotation: set ``ADMIN_PASSWORD`` and ``ADMIN_PASSWORD_FORCE_RESET=1``,
redeploy, then remove the flag.

It also provisions a MISSING administrator (work-orders #66): with
``ADMIN_EMAIL`` and ``ADMIN_PASSWORD`` set and no account for that address, the
administrator is created. An existing administrator is reset only with
``ADMIN_PASSWORD_FORCE_RESET=1``; an ordinary account is never promoted.
"""
import os
import sys


def main() -> int:
    from app import create_app
    app = create_app()
    if not app:
        print("default-admin guard: could not create the application "
              "(is DATABASE_URL set?) — nothing changed", file=sys.stderr)
        return 0

    from models import db
    from security.default_admin_guard import (
        STATUS_ABSENT, STATUS_FORCED, STATUS_NOT_DEFAULT, STATUS_REFUSED,
        STATUS_ROTATED, neutralise_default_admin)

    with app.app_context():
        try:
            result = neutralise_default_admin(
                db.session,
                replacement_password=os.environ.get('ADMIN_PASSWORD') or None,
                force=os.environ.get('ADMIN_PASSWORD_FORCE_RESET') == '1')
        except Exception as exc:  # noqa: BLE001 — report, never block the deploy
            db.session.rollback()
            print(f"default-admin guard: check failed ({exc.__class__.__name__}) "
                  "— nothing changed", file=sys.stderr)
            return 0

    _provision_admin(app, db)

    status = result['status']
    if status == STATUS_ABSENT:
        print(f"default-admin guard: no row for {result['email']} — nothing to do")
    elif status == STATUS_NOT_DEFAULT:
        print(f"default-admin guard: {result['email']} exists and is NOT on the "
              "default password — nothing changed")
    elif status == STATUS_ROTATED:
        how = ('ADMIN_PASSWORD from the environment' if result['used_env_password']
               else 'a random value — to recover, set ADMIN_PASSWORD and '
                    'ADMIN_PASSWORD_FORCE_RESET=1, redeploy, then remove the flag')
        print(f"default-admin guard: {result['email']} WAS on the default password "
              f"— rotated to {how}")
    elif status == STATUS_FORCED:
        print(f"default-admin guard: {result['email']} password reset to ADMIN_PASSWORD "
              "on request (ADMIN_PASSWORD_FORCE_RESET) — REMOVE the flag now")
    elif status == STATUS_REFUSED:
        # #14: exits NON-ZERO. The whole defect was a success report over a live
        # credential, so this must be impossible to mistake for "done" — including
        # by any automation that only checks the exit status.
        print(f"default-admin guard: REFUSED — ADMIN_PASSWORD is itself a known "
              f"default, so {result['email']} is STILL ON THE COMPROMISED PASSWORD. "
              f"Choose a different ADMIN_PASSWORD and run this again.",
              file=sys.stderr)
        return 1
    return 0


def _provision_admin(app, db) -> None:
    """Create the administrator named by ADMIN_EMAIL if it does not exist
    (work-orders #66). One status line; never a password; never blocks a deploy."""
    from security.default_admin_guard import (
        ADMIN_CREATED, ADMIN_EXISTS, ADMIN_FORCED, ADMIN_NOT_ADMIN, ADMIN_REFUSED,
        ADMIN_SKIPPED, ensure_admin_account)

    with app.app_context():
        try:
            result = ensure_admin_account(
                db.session,
                email=os.environ.get('ADMIN_EMAIL'),
                password=os.environ.get('ADMIN_PASSWORD'),
                username=os.environ.get('ADMIN_USERNAME'),
                force=os.environ.get('ADMIN_PASSWORD_FORCE_RESET') == '1')
        except Exception as exc:  # noqa: BLE001 — report, never block the deploy
            db.session.rollback()
            print(f"admin provisioning: check failed ({exc.__class__.__name__}) "
                  "— nothing changed", file=sys.stderr)
            return

    status, email = result['status'], result['email']
    if status == ADMIN_SKIPPED:
        print("admin provisioning: ADMIN_EMAIL and ADMIN_PASSWORD not both set — nothing to do")
    elif status == ADMIN_CREATED:
        print(f"admin provisioning: created administrator {email}")
    elif status == ADMIN_EXISTS:
        print(f"admin provisioning: administrator {email} exists — nothing changed")
    elif status == ADMIN_FORCED:
        print(f"admin provisioning: administrator {email} password reset to ADMIN_PASSWORD "
              "on request (ADMIN_PASSWORD_FORCE_RESET) — REMOVE the flag now")
    elif status == ADMIN_NOT_ADMIN:
        print(f"admin provisioning: REFUSED — {email} is an ordinary Analee account; it was "
              "not promoted or changed. Set ADMIN_EMAIL to an address with no account.",
              file=sys.stderr)
    elif status == ADMIN_REFUSED:
        print(f"admin provisioning: REFUSED — ADMIN_PASSWORD is a known default; "
              f"{email} was not created. Choose a different ADMIN_PASSWORD.",
              file=sys.stderr)


if __name__ == '__main__':
    sys.exit(main())
