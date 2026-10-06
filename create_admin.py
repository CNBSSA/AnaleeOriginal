"""
Create (or reset) the admin user from environment variables.

No credentials are hardcoded. Set these before running:

    ADMIN_EMAIL=you@example.com ADMIN_PASSWORD=YOUR_STRONG_PASSWORD \
        [ADMIN_USERNAME=Admin] python create_admin.py

This script goes through the same rules as the pre-deploy guard
(security/default_admin_guard.ensure_admin_account) — work-orders #69, R10:
a missing administrator is created; an EXISTING administrator's password is
reset to ADMIN_PASSWORD; a known-default ADMIN_PASSWORD is REFUSED; an address
that belongs to an ordinary (non-admin) account is NEVER promoted or reset.
Exit status is non-zero on a refusal. No password is ever printed.
"""
import os
import sys

from app import create_app
from models import db
from security.default_admin_guard import (
    ADMIN_CREATED, ADMIN_EXISTS, ADMIN_FORCED, ADMIN_NOT_ADMIN, ADMIN_REFUSED,
    ADMIN_SKIPPED, ensure_admin_account)


def create_admin_user():
    """Create the admin user if it doesn't exist, else reset its password —
    through the guard's refusals."""
    email = (os.environ.get('ADMIN_EMAIL') or '').lower().strip()
    password = os.environ.get('ADMIN_PASSWORD') or ''
    username = os.environ.get('ADMIN_USERNAME', 'Admin')

    if not email or not password:
        print("Set ADMIN_EMAIL and ADMIN_PASSWORD in the environment first.",
              file=sys.stderr)
        return 2

    app = create_app()
    if not app:
        print("Failed to create application (is DATABASE_URL set?).", file=sys.stderr)
        return 1

    with app.app_context():
        try:
            result = ensure_admin_account(
                db.session, email=email, password=password, username=username,
                force=True)
        except Exception as e:
            print(f"Error creating/updating admin user: {e.__class__.__name__}",
                  file=sys.stderr)
            db.session.rollback()
            return 1

    status = result['status']
    if status == ADMIN_CREATED:
        print(f"Admin user created: {email}")
        return 0
    if status in (ADMIN_FORCED, ADMIN_EXISTS):
        print(f"Admin user updated: {email}")
        return 0
    if status == ADMIN_NOT_ADMIN:
        print(f"REFUSED: {email} is an ordinary Analee account; it was not promoted "
              "or changed. Set ADMIN_EMAIL to an address with no account.",
              file=sys.stderr)
        return 1
    if status == ADMIN_REFUSED:
        print("REFUSED: ADMIN_PASSWORD is a known default; nothing was created or "
              "changed. Choose a different ADMIN_PASSWORD.", file=sys.stderr)
        return 1
    if status == ADMIN_SKIPPED:
        print("Set ADMIN_EMAIL and ADMIN_PASSWORD in the environment first.",
              file=sys.stderr)
        return 2
    print(f"Unexpected guard status: {status}", file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(create_admin_user())
