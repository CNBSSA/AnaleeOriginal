#!/usr/bin/env python3
"""One-time cleanup of R0.00 statement-notice rows already in the database.

Re-applies the Unit 1.7 junk filter to persisted ``Transaction`` rows (for
imports that landed before the filter existed). Only zero-amount rows matching
definite notice patterns are removed.

DRY-RUN by default — prints counts and sample descriptions. Pass ``--apply`` to
delete matching rows. Optional ``--user-id`` limits scope to one tenant.

    python scripts/cleanup_junk_transactions.py
    python scripts/cleanup_junk_transactions.py --user-id 42 --apply

Requires DATABASE_URL and the app's normal env. Use on staging / TEST databases
first; review the dry-run output before ``--apply``.
"""
from __future__ import annotations

import argparse
import json
import sys

from app import create_app
from models import db
from services.junk_transaction_cleanup import cleanup_definite_junk_transactions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete matching rows (default is dry-run only).",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        action="append",
        dest="user_ids",
        metavar="ID",
        help="Limit cleanup to one or more user ids (repeatable).",
    )
    args = parser.parse_args(argv)

    app = create_app()
    if not app:
        print("Could not create application (is DATABASE_URL set?).", file=sys.stderr)
        return 1

    with app.app_context():
        try:
            report = cleanup_definite_junk_transactions(
                db.session,
                user_ids=args.user_ids,
                apply=args.apply,
            )
        except Exception as exc:
            db.session.rollback()
            print(f"Cleanup failed: {exc}", file=sys.stderr)
            return 1

    payload = report.to_dict()
    payload["mode"] = "APPLIED" if args.apply else "DRY-RUN"
    print(json.dumps(payload, indent=2))
    if not args.apply and report.matched_junk:
        print(
            f"\nDRY-RUN: {report.matched_junk} row(s) would be deleted "
            f"(pass --apply to delete).",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
