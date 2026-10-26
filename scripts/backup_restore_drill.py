#!/usr/bin/env python3
"""Backup/restore drill — TEST data only, separate target database.

Proves a backup taken from SOURCE_DATABASE_URL restores into
RESTORE_TARGET_DATABASE_URL (must differ). Uses a throwaway marker table;
does not read or write production credentials beyond env URLs you supply.

Example (SQLite files under /tmp):
  SOURCE_DATABASE_URL=sqlite:////tmp/analee-drill-source.db \\
  RESTORE_TARGET_DATABASE_URL=sqlite:////tmp/analee-drill-restore.db \\
  python scripts/backup_restore_drill.py
"""
from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    source = os.environ.get('SOURCE_DATABASE_URL') or os.environ.get('DATABASE_URL')
    target = os.environ.get('RESTORE_TARGET_DATABASE_URL') or os.environ.get(
        'TEST_DATABASE_URL'
    )
    if not source:
        print('Set SOURCE_DATABASE_URL or DATABASE_URL', file=sys.stderr)
        return 2
    if not target:
        print('Set RESTORE_TARGET_DATABASE_URL or TEST_DATABASE_URL', file=sys.stderr)
        return 2
    if source == target:
        print('Refusing drill: source and target URLs must differ', file=sys.stderr)
        return 2

    from utils.backup_manager import run_restore_drill

    outcome = run_restore_drill(source, target)
    print(json.dumps(outcome, indent=2))
    return 0 if outcome.get('ok') else 1


if __name__ == '__main__':
    raise SystemExit(main())
