"""Main application configuration and initialization"""
import os
import logging
import sys
import tempfile
from datetime import datetime
from urllib.parse import urlparse
from flask import Flask, current_app, redirect, url_for, request, flash, jsonify
from flask_migrate import Migrate
from dotenv import load_dotenv
from sqlalchemy import text
from flask_apscheduler import APScheduler
from flask_wtf.csrf import CSRFProtect
from flask_login import LoginManager
from config import MAX_UPLOAD_BYTES

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask extensions
db = None  # Will be initialized with app context
migrate = Migrate()
scheduler = APScheduler()
csrf = CSRFProtect()
login_manager = LoginManager()


def _request_entity_too_large(error):
    """Friendly handler for 413 (request body over MAX_CONTENT_LENGTH).

    AJAX uploads (e.g. /upload, bank_statements) get a JSON error so their
    client-side handlers can show it; normal form posts (e.g. OCR uploads) get a
    flash and a redirect back to the page they came from instead of Werkzeug's
    default 413 page.
    """
    message = 'That upload is too large. Please choose a smaller file and try again.'
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
        return jsonify({'success': False, 'error': message}), 413

    flash(message, 'error')
    # Only redirect back to the referrer when it is a local URL (avoid open
    # redirects and 405s on POST-only endpoints); otherwise the upload page.
    referrer = request.referrer
    if referrer:
        parsed = urlparse(referrer)
        if not parsed.netloc or parsed.netloc == request.host:
            return redirect(referrer)
    return redirect(url_for('main.upload'))


def _resolve_secret_key():
    """Return a STABLE secret key for Flask sessions + CSRF.

    Why this matters: the signed session cookie holds the per-session CSRF
    token. If the key that signs it is not identical across every gunicorn
    worker, a form (and its CSRF token) rendered by one worker cannot be
    validated by another, and the user gets "Bad Request — The CSRF token is
    missing/invalid" on login, registration and EVERY upload. The Procfile runs
    gunicorn without --preload, so each worker would otherwise call
    ``os.urandom`` independently and end up with a different key.

    Resolution order:
      1. ``FLASK_SECRET_KEY`` from the environment — the correct production
         setup: identical across workers AND stable across restarts/redeploys.
      2. A key persisted to a file shared by all workers in this container, so
         that even when the env var is unset every worker agrees on one key.
         Created atomically (O_EXCL) so concurrent workers converge on a single
         value. (Rotates on redeploy — users log out on deploy — but CSRF works
         within a deploy, which is the acute failure being fixed.)
      3. A last-resort per-process random key, only if the file is unusable.
    """
    env_key = os.environ.get('FLASK_SECRET_KEY')
    if env_key:
        return env_key

    logger.warning(
        "FLASK_SECRET_KEY is NOT set — falling back to a key shared on disk so "
        "all workers agree (fixes CSRF), but it will NOT survive a redeploy. "
        "Set FLASK_SECRET_KEY in the environment for durable sessions.")
    print("[SECURITY] FLASK_SECRET_KEY not set — set it in Railway for durable "
          "sessions; using a shared on-disk fallback key for now.", flush=True)

    key_path = os.environ.get('SECRET_KEY_FILE') or os.path.join(
        tempfile.gettempdir(), 'analee_secret_key')

    def _read(path):
        try:
            with open(path, 'rb') as fh:
                data = fh.read()
                return data or None
        except OSError:
            return None

    # Fast path: another worker already created the shared key.
    existing = _read(key_path)
    if existing:
        return existing

    new_key = os.urandom(32)
    try:
        # Exclusive create: only the first worker writes; the rest fall through
        # to read the winner's key, so every worker ends up with the same value.
        fd = os.open(key_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(fd, new_key)
        finally:
            os.close(fd)
        return new_key
    except FileExistsError:
        return _read(key_path) or new_key
    except OSError as exc:
        logger.error(
            f"Could not persist fallback secret key ({exc}); using a "
            "per-process key — CSRF may still fail across workers.")
        return new_key


def create_app(env=None):
    """Create and configure the Flask application"""
    try:
        logger.info("Starting application creation...")
        global db

        # Initialize Flask application
        app = Flask(__name__,
                   template_folder='templates',
                   static_folder='static')

        # Get database URL
        database_url = os.environ.get('DATABASE_URL')
        if not database_url:
            logger.error("DATABASE_URL environment variable not set")
            raise ValueError("DATABASE_URL not configured")

        logger.info("Configuring application...")

        # SECRET_KEY must come from the environment for stable sessions. If it's
        # missing we still boot (don't take the app down) but warn LOUDLY: a random
        # per-boot key logs every user out on each redeploy and is inconsistent
        # across gunicorn workers. Set FLASK_SECRET_KEY to fix.
        secret_key = _resolve_secret_key()

        # Secure auth cookies in production (HTTPS). Gated so local HTTP dev still
        # works: ON when on Railway / FLASK_ENV=production, unless explicitly
        # overridden via SESSION_COOKIE_SECURE.
        _override = os.environ.get('SESSION_COOKIE_SECURE')
        if _override is not None:
            secure_cookies = _override.strip().lower() in ('1', 'true', 'yes', 'on')
        else:
            secure_cookies = bool(os.environ.get('RAILWAY_ENVIRONMENT')
                                  or os.environ.get('FLASK_ENV', '').lower() == 'production')

        # Configure Flask app
        app.config.update({
            'SECRET_KEY': secret_key,
            'SQLALCHEMY_DATABASE_URI': database_url,
            'SQLALCHEMY_TRACK_MODIFICATIONS': False,
            'TEMPLATES_AUTO_RELOAD': True,
            'MAX_CONTENT_LENGTH': MAX_UPLOAD_BYTES,
            'WTF_CSRF_ENABLED': True,
            'WTF_CSRF_TIME_LIMIT': 3600,
            'SESSION_COOKIE_SECURE': secure_cookies,
            'SESSION_COOKIE_HTTPONLY': True,
            'REMEMBER_COOKIE_SECURE': secure_cookies,
            'REMEMBER_COOKIE_HTTPONLY': True
        })

        # Import db after app creation to avoid circular imports
        from models import db as models_db, User
        global db
        db = models_db

        # Initialize extensions with app context
        db.init_app(app)
        migrate.init_app(app, db)
        csrf.init_app(app)

        # Configure login manager
        login_manager.init_app(app)
        login_manager.login_view = 'auth.login'
        login_manager.login_message = 'Please log in to access this page.'
        login_manager.login_message_category = 'info'
        login_manager.session_protection = 'strong'

        @login_manager.user_loader
        def load_user(user_id):
            """Load user by ID with enhanced error handling"""
            if not user_id:
                return None
            return User.query.get(int(user_id))

        # Import and register blueprints within app context
        with app.app_context():
            # Verify database connection
            db.session.execute(text('SELECT 1'))
            logger.info("Database connection verified")

            # Import blueprints
            from auth import auth
            from routes import main
            from historical_data import historical_data
            from bank_statements import bank_statements
            from chat import chat
            from reports import reports
            from admin import admin
            from risk_assessment import risk_assessment
            from recommendations import recommendations
            from predictions import predictions
            from suggestions import suggestions
            from errors import errors
            from ocr import ocr

            # Register blueprints
            app.register_blueprint(auth)
            app.register_blueprint(main)
            app.register_blueprint(historical_data)
            app.register_blueprint(bank_statements)
            app.register_blueprint(chat)
            app.register_blueprint(reports)
            app.register_blueprint(admin)
            app.register_blueprint(risk_assessment)
            app.register_blueprint(recommendations)
            app.register_blueprint(predictions)
            app.register_blueprint(suggestions)
            app.register_blueprint(errors)
            app.register_blueprint(ocr)

            # Friendly 413 handler for oversized uploads (MAX_CONTENT_LENGTH).
            app.register_error_handler(413, _request_entity_too_large)

            # Ensure database tables exist
            db.create_all()
            logger.info("Database tables verified")

            # Defensive schema guard: alert_history.alert_config_id was added to
            # the model after some databases were already created. create_all()
            # never ALTERs an existing table and this app does not run migrations
            # on deploy, so add the column if an existing alert_history table is
            # missing it. Wrapped so a failure can never block startup. (A proper
            # migration also exists for environments that run `flask db upgrade`.)
            try:
                from sqlalchemy import inspect as _sa_inspect, text as _sa_text
                _insp = _sa_inspect(db.engine)
                if 'alert_history' in _insp.get_table_names():
                    _cols = [c['name'] for c in _insp.get_columns('alert_history')]
                    if 'alert_config_id' not in _cols:
                        with db.engine.begin() as _conn:
                            _conn.execute(_sa_text(
                                'ALTER TABLE alert_history '
                                'ADD COLUMN alert_config_id INTEGER'
                            ))
                        logger.info("Added missing alert_history.alert_config_id column")
            except Exception as _e:
                logger.error(f"alert_history column guard skipped: {_e}")

            from services.entity_chart_schema import (
                ensure_company_settings_schema,
                ensure_entity_chart_schema,
            )
            if not ensure_company_settings_schema():
                logger.error(
                    'company_settings schema guard failed — settings pages may 500'
                )
            schema_ready = ensure_entity_chart_schema()
            if not schema_ready:
                logger.error(
                    'Entity chart schema guard failed — charts may be empty until fixed'
                )

            @app.cli.command('seed-charts')
            def seed_charts_command():
                """Seed entity types and master chart (BooksXperts parity)."""
                from services.chart_of_accounts import seed_entities, seed_admin_charts
                seed_entities()
                created, skipped = seed_admin_charts()
                print(f'Chart seed complete: {created} created, {skipped} skipped.')

            try:
                from services.chart_of_accounts import seed_entities, seed_admin_charts
                if schema_ready:
                    seed_entities()
                    created, skipped = seed_admin_charts()
                    logger.info(
                        'Chart seed on boot: %s created, %s skipped', created, skipped
                    )
            except Exception as chart_seed_exc:
                logger.error('Chart seed on boot failed: %s', chart_seed_exc)

            return app

    except Exception as e:
        logger.error(f"Critical error in application creation: {str(e)}")
        return None

def main():
    """Main entry point for the application"""
    try:
        app = create_app()
        if app:
            port = int(os.environ.get('PORT', 5001))
            app.run(host='0.0.0.0', port=port)
        else:
            logger.error("Application creation failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
