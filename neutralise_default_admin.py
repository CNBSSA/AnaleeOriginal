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


if __name__ == '__main__':
    sys.exit(main())
