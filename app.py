"""Main application configuration and initialization"""
import os
import logging
import sys
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
        secret_key = os.environ.get('FLASK_SECRET_KEY')
        if not secret_key:
            logger.warning(
                "FLASK_SECRET_KEY is NOT set — falling back to a random per-boot key. "
                "Sessions will NOT survive a restart/redeploy and will be inconsistent "
                "across workers. Set FLASK_SECRET_KEY in the environment to fix this.")
            print("[SECURITY] FLASK_SECRET_KEY not set — sessions won't persist across "
                  "redeploys; set FLASK_SECRET_KEY.", flush=True)
            secret_key = os.urandom(32)

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

            # Defensive schema guard: company_settings.logo / logo_type were added
            # to the model (the firm-logo feature) with an Alembic migration only.
            # This app builds its schema with create_all() and does NOT run
            # `flask db upgrade` on deploy, so on an existing database those columns
            # are missing and EVERY CompanySettings query (company settings page,
            # dashboard, reports, insights) fails with "column ... does not exist".
            # Add the columns if missing. Idempotent; never blocks startup.
            try:
                from sqlalchemy import inspect as _sa_inspect_cs, text as _sa_text_cs
                _insp_cs = _sa_inspect_cs(db.engine)
                if 'company_settings' in _insp_cs.get_table_names():
                    _cs_cols = [c['name'] for c in _insp_cs.get_columns('company_settings')]
                    _blob = 'BYTEA' if db.engine.dialect.name == 'postgresql' else 'BLOB'
                    _to_add = []
                    if 'logo' not in _cs_cols:
                        _to_add.append(('logo', _blob))
                    if 'logo_type' not in _cs_cols:
                        _to_add.append(('logo_type', 'VARCHAR(50)'))
                    if _to_add:
                        with db.engine.begin() as _conn_cs:
                            for _col, _coltype in _to_add:
                                _conn_cs.execute(_sa_text_cs(
                                    f'ALTER TABLE company_settings ADD COLUMN {_col} {_coltype}'
                                ))
                        logger.info(f"Added missing company_settings columns: {[c for c, _ in _to_add]}")
            except Exception as _e:
                logger.error(f"company_settings logo column guard skipped: {_e}")

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
