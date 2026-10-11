"""Backup → restore round-trip (SQLite file databases, TEST data only)."""
import os
from pathlib import Path

import pytest

from utils.backup_manager import DatabaseBackupManager, run_restore_drill


@pytest.fixture
def sqlite_pair(tmp_path, monkeypatch):
    """Two file-backed SQLite URLs and an isolated backup directory."""
    source = tmp_path / 'source.db'
    target = tmp_path / 'restore_target.db'
    backup_dir = tmp_path / 'backups'
    monkeypatch.setenv('ANALEE_BACKUP_DIR', str(backup_dir))
    source_url = f'sqlite:///{source}'
    target_url = f'sqlite:///{target}'
    return source_url, target_url, backup_dir


def test_sqlite_backup_lists_and_restores_to_separate_file(sqlite_pair):
    source_url, target_url, backup_dir = sqlite_pair
    from sqlalchemy import create_engine, text

    engine = create_engine(source_url)
    with engine.begin() as conn:
        conn.execute(
            text(
                'CREATE TABLE drill_payload (id INTEGER PRIMARY KEY, value TEXT NOT NULL)'
            )
        )
        conn.execute(text("INSERT INTO drill_payload (id, value) VALUES (1, 'round-trip')"))

    manager = DatabaseBackupManager(source_url)
    created = manager.create_backup()
    assert created is not None
    assert Path(created['file']).exists()
    assert list(manager.list_backups())

    outcome = run_restore_drill(
        source_url,
        target_url,
        marker_token='pytest-backup-marker',
    )
    assert outcome['ok'] is True

    restored = create_engine(target_url)
    with restored.connect() as conn:
        marker = conn.execute(
            text('SELECT token FROM backup_drill_marker WHERE id = 1')
        ).fetchone()
        payload = conn.execute(
            text('SELECT value FROM drill_payload WHERE id = 1')
        ).fetchone()
    assert marker[0] == 'pytest-backup-marker'
    assert payload[0] == 'round-trip'
    assert backup_dir.exists()


def test_restore_drill_refuses_same_database_url(sqlite_pair):
    source_url, _, _ = sqlite_pair
    with pytest.raises(ValueError, match='separate target'):
        run_restore_drill(source_url, source_url)


def test_backup_scheduler_dark_by_default():
    from utils.backup_manager import backup_scheduler_enabled

    os.environ.pop('ANALEE_BACKUP_SCHEDULER_ENABLED', None)
    assert backup_scheduler_enabled() is False
    os.environ['ANALEE_BACKUP_SCHEDULER_ENABLED'] = '1'
    assert backup_scheduler_enabled() is True
    os.environ.pop('ANALEE_BACKUP_SCHEDULER_ENABLED', None)
