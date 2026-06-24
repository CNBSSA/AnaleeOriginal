"""
Shared pytest fixtures for the canary smoke suite.

The full app stack (app.create_app) imports flask_migrate and flask_apscheduler.
Those are installed in CI/production but may be unbuildable in a minimal local
sandbox. When (and only when) they are genuinely absent, we install lightweight
stubs so the smoke suite can still boot the real app. In CI the real packages are
present and these stubs are never used.
"""
import importlib.util
import os
import sys
import tempfile
import types

import pytest


def _stub_if_missing(name, **attrs):
    try:
        if importlib.util.find_spec(name) is not None:
            return  # real package present (CI/prod) — use it
    except (ImportError, ValueError):
        pass
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


_stub_if_missing('flask_migrate', Migrate=type('Migrate', (), {
    '__init__': lambda self, *a, **k: None,
    'init_app': lambda self, *a, **k: None,
}))
_stub_if_missing('flask_apscheduler', APScheduler=type('APScheduler', (), {
    '__init__': lambda self, *a, **k: None,
    'init_app': lambda self, *a, **k: None,
    'start': lambda self, *a, **k: None,
}))


@pytest.fixture
def canary_app():
    """Boot the real application against a throwaway file-backed SQLite DB.

    A file (not :memory:) is used so every connection sees the same schema.
    CSRF is disabled for the test client and ANTHROPIC_API_KEY is removed so AI
    features take their deterministic offline fallback.
    """
    fd, db_path = tempfile.mkstemp(suffix='.db', prefix='canary_')
    os.close(fd)
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
    os.environ['FLASK_SECRET_KEY'] = 'canary-test-secret'
    os.environ.pop('ANTHROPIC_API_KEY', None)

    from app import create_app
    app = create_app()
    assert app is not None, "create_app() returned None"
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['TESTING'] = True

    yield app

    try:
        os.remove(db_path)
    except OSError:
        pass
