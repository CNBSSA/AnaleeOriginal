import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

from flask_apscheduler import APScheduler
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from config import env_flag

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_F = TypeVar('_F', bound=Callable[..., Any])


def _backup_dir() -> Path:
    raw = (os.environ.get('ANALEE_BACKUP_DIR') or 'backups').strip()
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_sqlite(database_url: str) -> bool:
    try:
        return make_url(database_url).drivername.startswith('sqlite')
    except Exception:
        return False


def _sqlite_database_path(database_url: str) -> Path:
    """Return filesystem path for a file-backed SQLite URL."""
    url = make_url(database_url)
    if url.database in (None, '', ':memory:'):
        raise ValueError('SQLite backup requires a file-backed database URL, not :memory:')
    db_path = Path(url.database)
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    return db_path


def _leader_locked(job: _F) -> _F:
    """Run at most one backup job at a time across gunicorn workers on one host."""

    @wraps(job)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        lock_path = _backup_dir() / '.scheduler.lock'
        try:
            import fcntl
        except ImportError:
            return job(*args, **kwargs)

        with open(lock_path, 'w') as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                logger.info('Backup job skipped: another worker holds the lock')
                return None
            return job(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


class DatabaseBackupManager:
    """Manages database backups and restorations (PostgreSQL pg_dump or SQLite file copy)."""

    def __init__(self, database_url: str, backup_dir: Optional[Path] = None):
        self.database_url = database_url
        self.backup_dir = backup_dir or _backup_dir()
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._sqlite = _is_sqlite(database_url)

        if self._sqlite:
            self.db_info = {'database': _sqlite_database_path(database_url).name}
            logger.info('Initialized SQLite backup manager for %s', self.db_info['database'])
        else:
            try:
                self.db_info = self._parse_database_url(database_url)
                logger.info(
                    'Initialized backup manager for database: %s', self.db_info['database']
                )
            except Exception as e:
                logger.error('Failed to parse database URL: %s', e)
                raise

    def _parse_database_url(self, url: str) -> Dict[str, str]:
        engine = create_engine(url)
        return {
            'host': engine.url.host or 'localhost',
            'port': str(engine.url.port or 5432),
            'user': engine.url.username or '',
            'password': engine.url.password or '',
            'database': engine.url.database or '',
        }

    def _metadata_path(self, timestamp: str) -> Path:
        return self.backup_dir / f'backup_{timestamp}_metadata.json'

    def _backup_path(self, timestamp: str) -> Path:
        ext = '.db' if self._sqlite else '.dump'
        return self.backup_dir / f'backup_{timestamp}{ext}'

    def create_backup(self) -> Optional[Dict[str, Any]]:
        """Create a new database backup."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self._backup_path(timestamp)

            if self._sqlite:
                src = _sqlite_database_path(self.database_url)
                if not src.exists():
                    logger.error('SQLite database file not found: %s', src)
                    return None
                shutil.copy2(src, backup_file)
            else:
                cmd = [
                    'pg_dump',
                    '--clean',
                    '--if-exists',
                    f"--host={self.db_info['host']}",
                    f"--port={self.db_info['port']}",
                    f"--username={self.db_info['user']}",
                    f"--dbname={self.db_info['database']}",
                    '--format=c',
                    '--file',
                    str(backup_file),
                ]
                env = os.environ.copy()
                env['PGPASSWORD'] = self.db_info['password']
                result = subprocess.run(cmd, env=env, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error('Backup failed: %s', result.stderr)
                    return None

            metadata = {
                'timestamp': timestamp,
                'database': self.db_info['database'],
                'size': os.path.getsize(backup_file),
                'engine': 'sqlite' if self._sqlite else 'postgresql',
                'source_url_scheme': make_url(self.database_url).drivername,
            }
            with open(self._metadata_path(timestamp), 'w') as f:
                json.dump(metadata, f)

            logger.info('Backup created successfully: %s', backup_file)
            return {'file': str(backup_file), 'metadata': metadata}
        except Exception as e:
            logger.error('Error creating backup: %s', e)
            return None

    def restore_backup_file(
        self,
        backup_file: str,
        target_database_url: Optional[str] = None,
    ) -> bool:
        """Restore a specific backup file into ``target_database_url`` (or the manager's DB)."""
        target_url = target_database_url or self.database_url
        path = Path(backup_file)
        if not path.exists():
            logger.error('Backup file not found: %s', path)
            return False

        try:
            if _is_sqlite(target_url):
                dest = _sqlite_database_path(target_url)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
                logger.info('SQLite restore completed into %s', dest)
                return True

            target_info = self._parse_database_url(target_url)
            cmd = [
                'pg_restore',
                '--clean',
                '--if-exists',
                f"--host={target_info['host']}",
                f"--port={target_info['port']}",
                f"--username={target_info['user']}",
                f"--dbname={target_info['database']}",
                str(path),
            ]
            env = os.environ.copy()
            env['PGPASSWORD'] = target_info['password']
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info('PostgreSQL restore completed into %s', target_info['database'])
                return True
            logger.error('Restore failed: %s', result.stderr)
            return False
        except Exception as e:
            logger.error('Error during restoration: %s', e)
            return False

    def restore_to_timestamp(self, target_timestamp: datetime) -> bool:
        backup = self._find_closest_backup(target_timestamp)
        if not backup:
            logger.error('No suitable backup found for restoration')
            return False
        return self.restore_backup_file(backup['file'])

    def _find_closest_backup(self, target_timestamp: datetime) -> Optional[Dict[str, Any]]:
        try:
            backups = self.list_backups()
            if not backups:
                return None

            closest_backup = None
            smallest_diff = timedelta.max

            for backup in backups:
                backup_time = datetime.strptime(
                    backup['metadata']['timestamp'],
                    '%Y%m%d_%H%M%S',
                )
                if backup_time <= target_timestamp:
                    diff = target_timestamp - backup_time
                    if diff < smallest_diff:
                        smallest_diff = diff
                        closest_backup = backup

            return closest_backup
        except Exception as e:
            logger.error('Error finding closest backup: %s', e)
            return None

    def restore_to_days_ago(self, days: int, target_minute: int = 59, target_hour: int = 23) -> bool:
        try:
            current_time = datetime.now()
            target_date = current_time - timedelta(days=days)
            target_timestamp = target_date.replace(
                hour=target_hour,
                minute=target_minute,
                second=59,
                microsecond=999999,
            )
            logger.info('Attempting to restore to %s', target_timestamp)
            return self.restore_to_timestamp(target_timestamp)
        except Exception as e:
            logger.error('Error during days-ago restoration: %s', e)
            return False

    def list_backups(self) -> List[Dict[str, Any]]:
        try:
            backups: List[Dict[str, Any]] = []
            for metadata_file in self.backup_dir.glob('backup_*_metadata.json'):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                timestamp = metadata['timestamp']
                for candidate in (
                    self.backup_dir / f'backup_{timestamp}.dump',
                    self.backup_dir / f'backup_{timestamp}.db',
                    self.backup_dir / f'backup_{timestamp}.sql',
                ):
                    if candidate.exists():
                        backups.append({'file': str(candidate), 'metadata': metadata})
                        break
            return sorted(backups, key=lambda x: x['metadata']['timestamp'])
        except Exception as e:
            logger.error('Error listing backups: %s', e)
            return []

    def cleanup_old_backups(self, keep_days: int = 7) -> None:
        try:
            cutoff_date = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)
            for backup in self.list_backups():
                backup_time = datetime.strptime(
                    backup['metadata']['timestamp'],
                    '%Y%m%d_%H%M%S',
                ).timestamp()
                if backup_time < cutoff_date:
                    backup_file = Path(backup['file'])
                    metadata_file = self._metadata_path(backup['metadata']['timestamp'])
                    if backup_file.exists():
                        backup_file.unlink()
                    if metadata_file.exists():
                        metadata_file.unlink()
                    logger.info('Removed old backup: %s', backup_file)
        except Exception as e:
            logger.error('Error cleaning up old backups: %s', e)


def run_restore_drill(
    source_database_url: str,
    restore_target_database_url: str,
    *,
    marker_token: str = 'analee-backup-drill-marker',
) -> Dict[str, Any]:
    """
    TEST / drill only: write a marker row, back up the source DB, restore into a
    separate target URL, and verify the marker is readable on the target.
    """
    if source_database_url == restore_target_database_url:
        raise ValueError('Restore drill requires a separate target database URL')

    src_engine = create_engine(source_database_url)
    with src_engine.begin() as conn:
        conn.execute(
            text(
                'CREATE TABLE IF NOT EXISTS backup_drill_marker '
                '(id INTEGER PRIMARY KEY, token TEXT NOT NULL)'
            )
        )
        conn.execute(text('DELETE FROM backup_drill_marker'))
        conn.execute(
            text('INSERT INTO backup_drill_marker (id, token) VALUES (1, :t)'),
            {'t': marker_token},
        )

    manager = DatabaseBackupManager(source_database_url)
    backup = manager.create_backup()
    if not backup:
        return {'ok': False, 'error': 'backup_failed'}

    target_path = Path(restore_target_database_url.replace('sqlite:///', '', 1))
    if target_path.exists():
        target_path.unlink()

    if not manager.restore_backup_file(backup['file'], restore_target_database_url):
        return {'ok': False, 'error': 'restore_failed', 'backup_file': backup['file']}

    tgt_engine = create_engine(restore_target_database_url)
    with tgt_engine.connect() as conn:
        row = conn.execute(
            text('SELECT token FROM backup_drill_marker WHERE id = 1')
        ).fetchone()
    token = row[0] if row else None
    ok = token == marker_token
    return {
        'ok': ok,
        'marker_token': marker_token,
        'restored_token': token,
        'backup_file': backup['file'],
        'source': source_database_url,
        'target': restore_target_database_url,
    }


def init_backup_scheduler(app, scheduler: Optional[APScheduler] = None) -> APScheduler:
    """
    Register daily backup + cleanup jobs on the given scheduler instance.

    Dark unless ``ANALEE_BACKUP_SCHEDULER_ENABLED`` is on (default off). Jobs are
    wrapped with a host-level file lock so multiple gunicorn workers do not run
    three copies of the same cron on one machine.
    """
    sched = scheduler or APScheduler()
    app.config.setdefault('SCHEDULER_API_ENABLED', False)
    app.config.setdefault(
        'SCHEDULER_EXECUTORS',
        {'default': {'type': 'threadpool', 'max_workers': 2}},
    )
    if 'apscheduler' not in app.extensions:
        sched.init_app(app)

    backup_manager = DatabaseBackupManager(app.config['SQLALCHEMY_DATABASE_URI'])

    @_leader_locked
    def _run_backup() -> Optional[Dict[str, Any]]:
        return backup_manager.create_backup()

    @_leader_locked
    def _run_cleanup() -> None:
        keep = int(os.environ.get('ANALEE_BACKUP_KEEP_DAYS', '7'))
        backup_manager.cleanup_old_backups(keep_days=keep)

    sched.add_job(
        id='daily_backup',
        func=_run_backup,
        trigger='cron',
        hour=int(os.environ.get('ANALEE_BACKUP_CRON_HOUR', '0')),
        minute=int(os.environ.get('ANALEE_BACKUP_CRON_MINUTE', '0')),
        replace_existing=True,
    )
    sched.add_job(
        id='backup_cleanup',
        func=_run_cleanup,
        trigger='cron',
        hour=int(os.environ.get('ANALEE_BACKUP_CLEANUP_HOUR', '1')),
        minute=int(os.environ.get('ANALEE_BACKUP_CLEANUP_MINUTE', '0')),
        replace_existing=True,
    )

    if not getattr(sched, 'running', False):
        if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
            sched.start()
    logger.info('Backup scheduler initialized (ANALEE_BACKUP_SCHEDULER_ENABLED)')
    return sched


def backup_scheduler_enabled() -> bool:
    return env_flag('ANALEE_BACKUP_SCHEDULER_ENABLED', default=False)
