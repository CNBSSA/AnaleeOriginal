"""Main application configuration and initialization"""
import os
import logging
import sys
import tempfile
from datetime import datetime
from urllib.parse import urlparse
from flask import Flask, current_app, redirect, url_for, request, flash, jsonify, session
from flask_migrate import Migrate
from dotenv import load_dotenv
from sqlalchemy import text
from flask_apscheduler import APScheduler
from flask_wtf.csrf import CSRFProtect
from flask_login import LoginManager, current_user
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

# Boot diagnostics surfaced by GET /health: what the schema-heal guard did at
# startup and when the app booted. Populated by create_app().
_BOOT_REPORT = {'booted_at': None, 'heal_added': [], 'heal_errors': []}

# Friendly 500 page. Deliberately inline HTML with ZERO dependencies (no
# templates, no DB, no session) so the handler itself can never fail.
_ERROR_500_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Something went wrong</title>
<style>body{font-family:system-ui,sans-serif;background:#f4f6f9;color:#1a1a1a;
margin:0;padding:2rem}main{max-width:560px;margin:8vh auto;background:#fff;
border-radius:14px;padding:2rem;box-shadow:0 2px 12px rgba(0,0,0,.06)}
h1{font-size:1.4rem;margin:0 0 .5rem}p{color:#5c6570;line-height:1.5}
a{color:#0d6efd}</style></head><body><main>
<h1>Something went wrong on our side</h1>
<p>The error has been recorded and we're on it. Please go back and try again
in a moment.</p>
<p><a href="/">Back to Analee</a></p>
</main></body></html>"""


def _internal_server_error(error):
    """Log every unhandled error with a searchable marker and show a friendly
    page instead of the bare Werkzeug 'Internal Server Error'. The traceback
    lands in the Railway logs (and Sentry when SENTRY_DSN is set) tagged
    [ANALEE-500] so it can be found instantly."""
    logger.exception("[ANALEE-500] Unhandled error on %s %s",
                     request.method, request.path)
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
        return jsonify({'success': False,
                        'error': 'Internal error — it has been logged.'}), 500
    return _ERROR_500_HTML, 500, {'Content-Type': 'text/html; charset=utf-8'}


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


def _heal_missing_columns(engine, models):
    """Add any model-defined columns that are missing from existing tables.

    This deploy builds its schema with ``create_all()`` and does NOT run
    ``flask db upgrade``. ``create_all()`` never ALTERs a table that already
    exists, so a column added to a model AFTER its table was first created is
    missing on the live database and every query on that table 500s. This guard
    closes that gap for the given models: it compares the model's columns to the
    live table and ``ALTER TABLE ... ADD COLUMN`` for anything missing.

    Safety:
      - Idempotent — a no-op when the column already exists.
      - Adds columns as NULLable with no default (never NOT NULL), so it is safe
        on a table that already has rows.
      - Skips primary keys and any column whose type can't be compiled.
      - Each ALTER is independent; one failure can't abort the others, and the
        whole helper never raises (callers stay safe at boot).

    Returns the list of ``(table, column)`` it added (useful for tests/logs).
    """
    from sqlalchemy import inspect as _inspect, text as _text
    added = []
    try:
        insp = _inspect(engine)
        tables = set(insp.get_table_names())
    except Exception as exc:
        logger.error(f"schema heal: could not inspect database: {exc}")
        return added
    for model in models:
        table = model.__tablename__
        if table not in tables:
            continue
        try:
            have = {c['name'] for c in insp.get_columns(table)}
        except Exception as exc:
            logger.error(f"schema heal: could not read columns of {table}: {exc}")
            continue
        for col in model.__table__.columns:
            if col.name in have or col.primary_key:
                continue
            try:
                ddl_type = col.type.compile(dialect=engine.dialect)
            except Exception as exc:
                logger.error(f"schema heal: skipping {table}.{col.name} "
                             f"(uncompilable type: {exc})")
                continue
            try:
                with engine.begin() as conn:
                    # Quote identifiers — e.g. "user" is a reserved word in Postgres.
                    conn.execute(_text(
                        f'ALTER TABLE "{table}" ADD COLUMN "{col.name}" {ddl_type}'
                    ))
                added.append((table, col.name))
                logger.info(f"schema heal: added {table}.{col.name} {ddl_type}")
            except Exception as exc:
                logger.error(f"schema heal: could not add {table}.{col.name}: {exc}")
    return added


def create_app(env=None):
    """Create and configure the Flask application"""
    try:
        logger.info("Starting application creation...")
        global db

        # Initialize Flask application
        app = Flask(__name__,
                   template_folder='templates',
                   static_folder='static')

        # Sentry error tracking (opt-in): active ONLY when SENTRY_DSN is set AND
        # sentry-sdk is installed. send_default_pii=False (no client PII). Errors
        # only. Wrapped so a bad/missing SDK never blocks startup.
        _sentry_dsn = os.environ.get('SENTRY_DSN', '')
        if _sentry_dsn:
            try:
                import sentry_sdk
                from sentry_sdk.integrations.flask import FlaskIntegration
                sentry_sdk.init(
                    dsn=_sentry_dsn,
                    integrations=[FlaskIntegration()],
                    environment=os.environ.get('ENV') or os.environ.get('FLASK_ENV', 'production'),
                    send_default_pii=False,
                    traces_sample_rate=0.0,
                )
            except Exception:
                pass

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

        @app.before_request
        def _analee_entitlement_gate():
            """Restrict Analee to Practice Club members OR Accountants/Analee
            subscribers (Festus 2026-07-13; scoped re-open of the frozen repo).

            Dark by default (``ANALEE_ENTITLEMENT_ENFORCED`` off) — no behaviour
            change. When enforced, an authenticated non-admin user who is neither
            a Club member (SSO session) nor a subscriber is redirected to a
            friendly notice; anonymous users are left to the login gate; admins
            are always allowed. Auth / static / error routes and the notice page
            itself are exempt so a blocked user can read it and log out.
            """
            import entitlement
            if not entitlement.enforcement_enabled():
                return
            if not current_user.is_authenticated:
                return
            if getattr(current_user, "is_admin", False):
                return
            endpoint = request.endpoint or ""
            if (endpoint in ("main.entitlement_required", "static")
                    or endpoint.startswith("auth.")
                    or endpoint.startswith("errors.")
                    or endpoint.startswith("provisioning.")):
                return
            if not entitlement.analee_entitled(current_user):
                return redirect(url_for("main.entitlement_required"))

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
            from provisioning import provisioning

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
            app.register_blueprint(provisioning)
            # Server-to-server provisioning endpoint uses a bearer secret, not a
            # browser session — exempt it from CSRF (it never handles form posts).
            csrf.exempt(provisioning)
            from client_explain_routes import client_explain_bp
            app.register_blueprint(client_explain_bp)
            # The client-explain wizard is a NO-LOGIN flow: clients have no session
            # and the form carries no CSRF token, so app-wide CSRF would reject
            # every save POST ("CSRF token is missing"). Security is provided by the
            # signed, scoped, expiring token + per-transaction ownership re-checks,
            # so exempt only this blueprint from CSRF (nothing else is affected).
            csrf.exempt(client_explain_bp)

            # Friendly 413 handler for oversized uploads (MAX_CONTENT_LENGTH).
            app.register_error_handler(413, _request_entity_too_large)

            # Ensure database tables exist
            db.create_all()
            logger.info("Database tables verified")

            # Defensive schema guard: heal any model columns missing from the
            # live `user` / `account` / `transaction` tables before anything
            # queries or seeds them. These are on the auth + OCR/bank-statement
            # upload + client-explain paths, so a drifted production DB (deploy
            # uses create_all(), not migrations) would otherwise 500 those pages.
            # `transaction` is included because `explanation_source` (added with
            # the client-explain feature via migration only) is a NOT-NULL mapped
            # column — without it EVERY Transaction query 500s in production.
            # Idempotent; never blocks startup.
            try:
                from models import (User as _User_heal, Account as _Account_heal,
                                    Transaction as _Txn_heal)
                _BOOT_REPORT['heal_added'] = [
                    f"{t}.{c}" for t, c in
                    _heal_missing_columns(db.engine, [_User_heal, _Account_heal, _Txn_heal])
                ]
            except Exception as _e:
                logger.error(f"schema heal guard skipped: {_e}")
                _BOOT_REPORT['heal_errors'].append(str(_e))

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

            # The Practice Club SSO consumer — SEALED module (scoped unfreeze,
            # Festus 2026-07-10; see CLAUDE.md). Dark unless CLUB_ENABLED is
            # set; register() is fail-soft and can never block startup.
            import club_sso
            club_sso.register(app)

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

            # Friendly 500 page + [ANALEE-500]-tagged traceback in the logs,
            # replacing the bare Werkzeug "Internal Server Error".
            app.register_error_handler(500, _internal_server_error)

            _BOOT_REPORT['booted_at'] = datetime.utcnow().isoformat() + 'Z'

            @app.route('/health')
            def _health():
                """No-login diagnostics: which build is live and is it healthy.

                Exposes ONLY metadata — a short commit id, booleans, and
                schema-column names. Never secret values, never user data.
                Lets anyone verify in one browser visit: (1) the running git
                commit (stale-deploy detection), (2) DB connectivity, (3) that
                user/account/transaction carry every model column (schema
                drift), (4) which required env vars are configured.
                """
                from sqlalchemy import inspect as _hins, text as _htext
                from models import User as _HU, Account as _HA, Transaction as _HT

                report = {
                    'commit': (os.environ.get('RAILWAY_GIT_COMMIT_SHA')
                               or 'unknown')[:12],
                    'booted_at': _BOOT_REPORT['booted_at'],
                    'boot_guard': {'added': _BOOT_REPORT['heal_added'],
                                   'errors': _BOOT_REPORT['heal_errors']},
                    'env': {k: bool(os.environ.get(k)) for k in
                            ('FLASK_SECRET_KEY', 'ANTHROPIC_API_KEY',
                             'SENTRY_DSN', 'DATABASE_URL')},
                }
                healthy = True
                try:
                    db.session.execute(_htext('SELECT 1'))
                    report['db_ok'] = True
                except Exception as exc:
                    report['db_ok'] = False
                    report['db_error'] = type(exc).__name__
                    healthy = False
                    try:
                        db.session.rollback()
                    except Exception:
                        pass

                missing = {}
                if report['db_ok']:
                    try:
                        insp = _hins(db.engine)
                        live_tables = set(insp.get_table_names())
                        for model in (_HU, _HA, _HT):
                            table = model.__tablename__
                            if table not in live_tables:
                                missing[table] = ['<table missing>']
                                continue
                            have = {c['name'] for c in insp.get_columns(table)}
                            gap = sorted(c.name for c in model.__table__.columns
                                         if c.name not in have)
                            if gap:
                                missing[table] = gap
                    except Exception as exc:
                        missing['<inspect_error>'] = [type(exc).__name__]
                if missing:
                    healthy = False
                report['schema_missing'] = missing

                # A BuildError on any nav endpoint breaks every page render.
                broken = []
                for endpoint in ('main.upload', 'main.analyze_list',
                                 'ocr.upload_statement', 'main.company_settings',
                                 'reports.trial_balance', 'auth.login'):
                    try:
                        url_for(endpoint)
                    except Exception as exc:
                        broken.append(f'{endpoint}: {type(exc).__name__}')
                if broken:
                    healthy = False
                    report['endpoint_errors'] = broken

                report['status'] = 'ok' if healthy else 'degraded'
                return jsonify(report), (200 if healthy else 503)

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
