import os
import logging
import sys
from datetime import datetime
from flask import Flask
from flask_migrate import Migrate
from dotenv import load_dotenv
from sqlalchemy import text
from flask_apscheduler import APScheduler
from models import db, login_manager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize Flask extensions
migrate = Migrate()
scheduler = APScheduler()

# Configure APScheduler
class Config:
    SCHEDULER_API_ENABLED = True
    SCHEDULER_EXECUTORS = {
        'default': {'type': 'threadpool', 'max_workers': 20}
    }
    SCHEDULER_JOB_DEFAULTS = {
        'coalesce': False,
        'max_instances': 3
    }

def verify_database():
    """Verify database connection"""
    try:
        logger.info("Verifying database connection...")
        with db.engine.connect() as conn:
            conn.execute(text('SELECT 1'))
        logger.info("Database connection verified")
        return True
    except Exception as db_error:
        logger.error(f"Database connection failed: {str(db_error)}")
        if current_app.debug:
            logger.exception("Full database connection error:")
        return False

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    logger.info("Starting Flask application initialization...")
    
    # Get database URL and handle legacy format
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable is not set")
        raise ValueError("DATABASE_URL environment variable is not set")
    
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
        
    # Configure Flask app
    app.config.update(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", os.urandom(24).hex()),
        SQLALCHEMY_DATABASE_URI=database_url,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        TEMPLATES_AUTO_RELOAD=True,
        DEBUG=True,
        RATELIMIT_DEFAULT="100 per minute",
        RATELIMIT_STORAGE_URL=database_url,
        RATELIMIT_STRATEGY='fixed-window',
        RATELIMIT_KEY_PREFIX='global_',
        RATELIMIT_HEADERS_ENABLED=True,
        SCHEDULER_EXECUTORS={'default': {'type': 'threadpool', 'max_workers': 20}},
        SCHEDULER_JOB_DEFAULTS={'coalesce': False, 'max_instances': 3}
    )
    logger.debug("Flask app configuration completed")

    # Initialize Flask extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    login_manager.login_view = 'main.login'
    
    # Import user loader here to avoid circular imports
    from models import load_user
    login_manager.user_loader(load_user)
    
    # Initialize scheduler
    scheduler.init_app(app)
    if not scheduler.running:
        scheduler.start()
        logger.debug("Scheduler started successfully")
    
    with app.app_context():
        # Import models here to avoid circular imports
        from models import User, Account, Transaction, UploadedFile, CompanySettings
        
        # Verify database connection
        try:
            db.session.execute(text('SELECT 1'))
            logger.info("Database connection verified")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise
        
        # Create database tables
        db.create_all()
        logger.info("Database tables created")
        
        # Initialize test suite after database is ready
        try:
            from tests.rollback_verification import RollbackVerificationTest
            if not hasattr(app, 'rollback_verification'):
                app.rollback_verification = RollbackVerificationTest(app)
                logger.info("Rollback verification test suite initialized")
        except ImportError as e:
            logger.warning(f"Test suite module not found (non-critical): {str(e)}")
        except Exception as e:
            logger.warning(f"Test suite initialization deferred (non-critical): {str(e)}")
        
        # Register blueprints
        try:
            from routes import main as main_blueprint
            app.register_blueprint(main_blueprint)
            logger.debug("Blueprints registered successfully")
        except Exception as blueprint_error:
            logger.error(f"Error registering blueprints: {str(blueprint_error)}")
            raise
        
        logger.info("Application initialization completed successfully")
        return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000)