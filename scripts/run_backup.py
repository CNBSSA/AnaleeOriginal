#!/usr/bin/env python3
"""Create a database backup (PostgreSQL pg_dump or SQLite file copy).

Use from a cron job or Railway scheduled task instead of relying on in-process
APScheduler when you want exactly one backup per schedule across replicas.

Requires DATABASE_URL. Writes under ANALEE_BACKUP_DIR (default: ./backups).
"""
from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print('DATABASE_URL is not set', file=sys.stderr)
        return 2

    from utils.backup_manager import DatabaseBackupManager

    manager = DatabaseBackupManager(database_url)
    result = manager.create_backup()
    if not result:
        print('Backup failed', file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
