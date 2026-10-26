# Backup and restore runbook (Analee)

This runbook covers **TEST and operational** backup/restore for the Analee
standalone app. It does not use production customer data in examples.

## How backups are taken

| Engine | Mechanism | Artifact |
|--------|-----------|----------|
| PostgreSQL (Railway) | `pg_dump` custom format | `backups/backup_YYYYMMDD_HHMMSS.dump` + JSON metadata |
| SQLite (local/CI) | File copy of the `.db` file | `backups/backup_YYYYMMDD_HHMMSS.db` + JSON metadata |

Implementation: `utils/backup_manager.py`  
CLI (preferred for a single scheduled job): `scripts/run_backup.py`

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | (required) | Database to back up |
| `ANALEE_BACKUP_DIR` | `backups` | Directory for artifacts (use a mounted volume on Railway) |
| `ANALEE_BACKUP_KEEP_DAYS` | `7` | Retention for scheduled cleanup |
| `ANALEE_BACKUP_SCHEDULER_ENABLED` | off | In-app daily cron (see below) |
| `ANALEE_BACKUP_CRON_HOUR` / `MINUTE` | `0` / `0` | Daily backup time (UTC) |
| `ANALEE_BACKUP_CLEANUP_HOUR` / `MINUTE` | `1` / `0` | Daily cleanup time (UTC) |

## Scheduled backups

**Recommended:** run `python scripts/run_backup.py` once per day from Railway
cron, GitHub Actions, or another external scheduler. That guarantees one backup
per run even with multiple app replicas.

**Optional in-process scheduler:** set `ANALEE_BACKUP_SCHEDULER_ENABLED=1` on
**one** web service (or accept that a file lock limits duplicate runs per host).
Jobs use a lock file under `ANALEE_BACKUP_DIR` so multiple gunicorn workers on
the same machine do not all fire `pg_dump` at midnight.

## Manual backup

```bash
export DATABASE_URL='postgresql://...'   # or sqlite:///path/to/file.db
export ANALEE_BACKUP_DIR=/var/data/backups   # durable path on the host/volume
python scripts/run_backup.py
```

Stdout is JSON with `file` and `metadata` on success; exit code `0`.

## Restore drill (separate database — TEST only)

Never restore into the live production URL without an explicit maintenance
window and a fresh backup. For drills, use a **different** URL.

```bash
export SOURCE_DATABASE_URL='sqlite:////tmp/analee-drill-source.db'
export RESTORE_TARGET_DATABASE_URL='sqlite:////tmp/analee-drill-restore.db'
python scripts/backup_restore_drill.py
```

The drill:

1. Writes a marker row into `backup_drill_marker` on the source.
2. Takes a backup via `DatabaseBackupManager.create_backup()`.
3. Restores into the target URL.
4. Verifies the marker (and exits non-zero on failure).

For PostgreSQL, point both URLs at empty **test** databases on the same cluster
(e.g. `analee_drill_source` / `analee_drill_restore`). `pg_restore --clean` is
used; the target database must exist and be disposable.

### Restore a specific file (PostgreSQL)

```bash
export PGPASSWORD='...'
pg_restore --clean --if-exists \
  --host=HOST --port=5432 --username=USER \
  --dbname=TARGET_DB \
  backups/backup_YYYYMMDD_HHMMSS.dump
```

### Restore a specific file (SQLite)

```bash
cp backups/backup_YYYYMMDD_HHMMSS.db /path/to/restored.db
# Point DATABASE_URL at the restored file for a throwaway instance.
```

## Automated proof in CI

`tests/test_backup_restore.py` runs a full SQLite backup → restore round-trip
using temporary files only (no production credentials).

## Evidence checklist

- [ ] `python scripts/run_backup.py` exits 0 and creates `backup_*` under `ANALEE_BACKUP_DIR`
- [ ] `python scripts/backup_restore_drill.py` exits 0 with `"ok": true` against test URLs
- [ ] `pytest tests/test_backup_restore.py` passes in CI
- [ ] Backup artifacts stored on durable storage (volume/S3), not ephemeral container disk only
