"""ProductionConfig's session settings are applied in production (work-orders #69, R7).

config.py carried a ProductionConfig (secure cookies, a 30-minute
PERMANENT_SESSION_LIFETIME) that create_app never read — it imported only three
constants — so the 30-minute lifetime did not exist and SESSION_COOKIE_SAMESITE
was unset. Now: SameSite=Lax on every environment, and in production the
lifetime is ProductionConfig's value.

Stated plainly: the app never marks a session permanent, so the lifetime governs
no live cookie yet — nobody is logged out by this change. Turning it into an
idle timeout is a user-facing decision for Festus. Remember-me duration stays at
Flask-Login's default.
"""
import os
import tempfile
from datetime import timedelta

import pytest

pytest.importorskip("flask_sqlalchemy")


def _boot(monkeypatch, production):
    fd, path = tempfile.mkstemp(suffix=".db", prefix="cfg_")
    os.close(fd)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    monkeypatch.setenv("FLASK_SECRET_KEY", "config-test-secret")
    monkeypatch.delenv("SESSION_COOKIE_SECURE", raising=False)
    if production:
        monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    else:
        monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
        monkeypatch.setenv("FLASK_ENV", "development")
    from app import create_app
    app = create_app()
    assert app is not None
    return app


def test_samesite_lax_everywhere(monkeypatch):
    app = _boot(monkeypatch, production=False)
    assert app.config["SESSION_COOKIE_SAMESITE"] == "Lax"
    client = app.test_client()
    client.get("/auth/login")
    cookie = next((c for c in client._cookies.values()
                   if c.key == app.config["SESSION_COOKIE_NAME"]), None)
    assert cookie is not None, "the login page sets a session cookie (CSRF token)"
    assert cookie.same_site == "Lax"


def test_production_applies_the_thirty_minute_lifetime(monkeypatch):
    from config import ProductionConfig
    app = _boot(monkeypatch, production=True)
    assert app.config["PERMANENT_SESSION_LIFETIME"] == timedelta(
        seconds=ProductionConfig.PERMANENT_SESSION_LIFETIME)
    assert app.config["SESSION_COOKIE_SECURE"] is True
    assert app.config["SESSION_COOKIE_HTTPONLY"] is True
    assert app.config["SESSION_COOKIE_SAMESITE"] == "Lax"


def test_development_keeps_flasks_default_lifetime(monkeypatch):
    app = _boot(monkeypatch, production=False)
    assert app.config["PERMANENT_SESSION_LIFETIME"] == timedelta(days=31)
