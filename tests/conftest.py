"""Pytest fixtures for AnaleeOriginal."""
import os

import pytest
from flask import Flask

os.environ.setdefault('FLASK_SECRET_KEY', 'test-secret-key')
os.environ.setdefault('DATABASE_URL', 'sqlite:///:memory:')

from models import db, User  # noqa: E402


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
