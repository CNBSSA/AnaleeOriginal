"""
Shared pytest fixtures for AnaleeOriginal.

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
from flask import Flask


def _stub_if_missing(name, **attrs):
    try:
        if importlib.util.find_spec(name) is not None:
            return
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

os.environ.setdefault('FLASK_SECRET_KEY', 'test-secret-key')
os.environ.setdefault('DATABASE_URL', 'sqlite:///:memory:')

from models import db, User  # noqa: E402


@pytest.fixture
def canary_app():
    """Boot the real application against a throwaway file-backed SQLite DB."""
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


@pytest.fixture
def app():
    """Minimal Flask app with in-memory SQLite for service-layer tests."""
    flask_app = Flask(__name__)
    flask_app.config.update({
        'TESTING': True,
        'SECRET_KEY': 'test-secret-key',
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'SQLALCHEMY_TRACK_MODIFICATIONS': False,
        'WTF_CSRF_ENABLED': False,
    })
    db.init_app(flask_app)
    with flask_app.app_context():
        db.create_all()
        yield flask_app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def sample_user(app):
    """Subscriber user for chart provisioning tests."""
    with app.app_context():
        user = User(
            username='testuser',
            email='test@example.com',
            subscription_status='active',
        )
        user.set_password('password')
        db.session.add(user)
        db.session.commit()
        user_id = user.id
    return user_id
