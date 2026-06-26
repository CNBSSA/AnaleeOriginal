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
from flask.json.provider import DefaultJSONProvider
from decimal import Decimal
from config import MAX_UPLOAD_BYTES


class _MoneyJSONProvider(DefaultJSONProvider):
    """JSON provider that serialises Decimal (money columns are Numeric/Decimal).

    Flask's default provider raises on Decimal; emit it as a JSON number so the
    existing API/AJAX responses keep working unchanged.
    """

    @staticmethod
    def default(o):
        if isinstance(o, Decimal):
            return float(o)
        return DefaultJSONProvider.default(o)

# Configure logging with detailed format.
# Log to stdout only: the platform (Railway) captures stdout, and a file handler
# would write to the ephemeral container filesystem (lost on every restart).
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
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
        app.json = _MoneyJSONProvider(app)  # serialise Decimal money values

        # Get database URL
        database_url = os.environ.get('DATABASE_URL')
        if not database_url:
            logger.error("DATABASE_URL environment variable not set")
            raise ValueError("DATABASE_URL not configured")

        logger.info("Configuring application...")

        # Configure Flask app
        app.config.update({
            'SECRET_KEY': os.environ.get('FLASK_SECRET_KEY', os.urandom(32)),
            'SQLALCHEMY_DATABASE_URI': database_url,
            'SQLALCHEMY_TRACK_MODIFICATIONS': False,
            'TEMPLATES_AUTO_RELOAD': True,
            'MAX_CONTENT_LENGTH': MAX_UPLOAD_BYTES,
            'WTF_CSRF_ENABLED': True,
            'WTF_CSRF_TIME_LIMIT': 3600,
            'SESSION_COOKIE_SECURE': False,
            'SESSION_COOKIE_HTTPONLY': True,
            'REMEMBER_COOKIE_SECURE': False,
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

            # Money-precision guard: the currency columns were originally Float
            # (double precision). create_all() never changes an existing column's
            # type, so convert them to NUMERIC(18,2) on existing PostgreSQL
            # databases — money must be exact to the cent. (SQLite has loose type
            # affinity and the ORM already returns Decimal, so it is skipped.)
            # Idempotent; wrapped so it can never block startup.
            try:
                from sqlalchemy import inspect as _sa_inspect2, text as _sa_text2
                if db.engine.dialect.name == 'postgresql':
                    _money_cols = [
                        ('transaction', 'amount'),
                        ('historical_data', 'amount'),
                        ('financial_goal', 'target_amount'),
                        ('financial_goal', 'current_amount'),
                    ]
                    _insp2 = _sa_inspect2(db.engine)
                    _tables = set(_insp2.get_table_names())
                    with db.engine.begin() as _conn2:
                        for _table, _col in _money_cols:
                            if _table not in _tables:
                                continue
                            _ctype = next((c['type'] for c in _insp2.get_columns(_table)
                                           if c['name'] == _col), None)
                            if _ctype is not None and 'NUMERIC' not in str(_ctype).upper():
                                _conn2.execute(_sa_text2(
                                    f'ALTER TABLE "{_table}" ALTER COLUMN {_col} '
                                    f'TYPE NUMERIC(18, 2) USING {_col}::numeric(18, 2)'
                                ))
                                logger.info(f"Converted {_table}.{_col} to NUMERIC(18, 2)")
            except Exception as _e:
                logger.error(f"money precision guard skipped: {_e}")

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
