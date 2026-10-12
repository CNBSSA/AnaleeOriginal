#!/usr/bin/env python3
"""Remove persisted zero-amount extraction junk (pre–Unit 1.7 imports). TEST data only.

Dry-run by default. To delete rows, set ANALEE_EXTRACTION_JUNK_CLEANUP=1 and pass --apply.

Example:
  python scripts/cleanup_extraction_junk_transactions.py
  ANALEE_EXTRACTION_JUNK_CLEANUP=1 python scripts/cleanup_extraction_junk_transactions.py --apply
  python scripts/cleanup_extraction_junk_transactions.py --user-id 42 --file-id 7
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete matching rows (default is dry-run only).",
    )
    parser.add_argument("--user-id", type=int, default=None, help="Limit to one user.")
    parser.add_argument("--file-id", type=int, default=None, help="Limit to one statement file.")
    args = parser.parse_args()

    dry_run = not args.apply
    if args.apply and os.environ.get("ANALEE_EXTRACTION_JUNK_CLEANUP") != "1":
        print(
            "Refusing --apply: set ANALEE_EXTRACTION_JUNK_CLEANUP=1 (TEST data only).",
            file=sys.stderr,
        )
        return 2

    from app import create_app
    from services.extraction_junk_cleanup import cleanup_junk_transactions, sample_matched_descriptions

    app = create_app()
    if not app:
        print("Failed to create application (is DATABASE_URL set?).", file=sys.stderr)
        return 1

    with app.app_context():
        stats = cleanup_junk_transactions(
            dry_run=dry_run,
            user_id=args.user_id,
            file_id=args.file_id,
        )
        preview = sample_matched_descriptions(
            user_id=args.user_id,
            file_id=args.file_id,
        )
        payload = {**stats.as_dict(), "sample_descriptions": list(preview)}
        print(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
