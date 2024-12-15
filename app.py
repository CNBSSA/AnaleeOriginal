import os
import logging
import sys
from datetime import datetime
from flask import Flask, current_app
from flask_migrate import Migrate
from dotenv import load_dotenv
from sqlalchemy import text
from flask_apscheduler import APScheduler
from models import db, login_manager

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Create module logger
logger = logging.getLogger(__name__)

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

def create_app(env='production'):
    """Create and configure the Flask application"""
    try:
        # Initialize Flask application
        app = Flask(__name__)
        logger.info(f"Starting Flask application initialization in {env} environment...")
        
        # Get database URL with enhanced error handling
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            logger.error("DATABASE_URL environment variable is not set")
            return None
            
        # Handle legacy database URL format
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
            logger.info("Converted legacy postgres:// URL format to postgresql://")
            
        # Configure logging for database operations with proper level
        db_logger = logging.getLogger('sqlalchemy.engine')
        db_logger.setLevel(logging.INFO)
        
        # Verify database URL format
        if not any(database_url.startswith(prefix) for prefix in ['postgresql://', 'postgres://']):
            logger.error(f"Invalid database URL format")
            return None
            
        # Basic application configuration check
        if not app.config:
            logger.error("Flask app configuration missing")
            return None
            
        # Configure Flask app with enhanced error handling
        config = {
            'ENV': env,
            'SECRET_KEY': os.environ.get("FLASK_SECRET_KEY", os.urandom(24).hex()),
            'SQLALCHEMY_TRACK_MODIFICATIONS': False,
            'SQLALCHEMY_DATABASE_URI': database_url,
            'SQLALCHEMY_ECHO': True,  # Enable SQL query logging
            'TEMPLATES_AUTO_RELOAD': True,
            'RATELIMIT_DEFAULT': "100 per minute",
            'RATELIMIT_STRATEGY': 'fixed-window',
            'RATELIMIT_KEY_PREFIX': 'global_',
            'RATELIMIT_HEADERS_ENABLED': True,
            'SCHEDULER_EXECUTORS': {'default': {'type': 'threadpool', 'max_workers': 20}},
            'SCHEDULER_JOB_DEFAULTS': {'coalesce': False, 'max_instances': 3}
        }
    
    # Environment-specific configuration
        if env == 'testing':
            config.update({
                'TESTING': True,
                'DEBUG': True,
                'ENABLE_ROLLBACK_TESTS': True,
                'SQLALCHEMY_DATABASE_URI': os.environ.get('TEST_DATABASE_URL', f"{database_url}_test"),
                'RATELIMIT_STORAGE_URL': os.environ.get('TEST_DATABASE_URL', f"{database_url}_test")
            })
        else:
            config.update({
                'TESTING': False,
                'DEBUG': False,
                'ENABLE_ROLLBACK_TESTS': False,
                'SQLALCHEMY_DATABASE_URI': database_url,
                'RATELIMIT_STORAGE_URL': database_url
            })
        
        app.config.update(config)
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
            try:
                # Verify database connection with timeout
                try:
                    db.session.execute(text('SELECT 1'))
                    logger.info("Database connection verified")
                except Exception as db_error:
                    logger.error(f"Database connection failed: {str(db_error)}")
                    return None
                
                # Import models with error handling
                try:
                    from models import User, Account, Transaction, UploadedFile, CompanySettings
                    logger.debug("Models imported successfully")
                except ImportError as import_error:
                    logger.error(f"Failed to import models: {str(import_error)}")
                    return None
                
                # Create database tables with proper error handling
                try:
                    db.create_all()
                    logger.info("Database tables created successfully")
                except Exception as table_error:
                    logger.error(f"Failed to create database tables: {str(table_error)}")
                    return None
                
                # Initialize test suite (non-critical)
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
                
            except Exception as e:
                logger.error(f"Error during application initialization: {str(e)}")
                raise
                
    except Exception as e:
        logger.error(f"Critical error in application creation: {str(e)}")
        return None

if __name__ == '__main__':
    try:
        # Initialize the application
        logger.info("Starting application initialization...")
        app = create_app()
        
        if not app:
            raise ValueError("Application creation failed")
            
        # Get port and configure server
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"Configuring server to run on port {port}")
        
        # Verify database connection before starting
        with app.app_context():
            db.session.execute(text('SELECT 1'))
            logger.info("Database connection verified")
            
        # Start the server
        logger.info(f"Starting Flask application on http://0.0.0.0:{port}")
        app.run(
            host='0.0.0.0',
            port=port,
            debug=True,
            use_reloader=False  # Disable reloader to prevent duplicate processes
        )
    except Exception as e:
        logger.error(f"Critical error starting application: {str(e)}")
        logger.exception("Full stack trace:")
        sys.exit(1)