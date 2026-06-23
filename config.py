import os
from dotenv import load_dotenv

load_dotenv()

# Single source of truth for the Claude model id used by every AI feature.
# Override per-environment with the CLAUDE_MODEL env var, no code change needed.
CLAUDE_MODEL = os.environ.get('CLAUDE_MODEL', 'claude-opus-4-7')

# Global ceiling on request body size (Flask MAX_CONTENT_LENGTH). Werkzeug rejects
# larger requests with 413 before reading the body into memory, guarding against
# memory exhaustion from oversized uploads. Must stay above the largest legitimate
# upload — the OCR PDF statement limit (ocr.service.MAX_PDF_BYTES = 32 MB) — so 40 MB
# leaves headroom for multipart overhead. Per-route checks give friendlier messages
# under this ceiling. Override with MAX_UPLOAD_MB.
MAX_UPLOAD_BYTES = int(os.environ.get('MAX_UPLOAD_MB', '40')) * 1024 * 1024

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', os.urandom(24).hex())
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    TEMPLATES_AUTO_RELOAD = True
    PROTECT_PRODUCTION = True  # Global flag to prevent production modifications

    # Email configuration for password reset
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', True)
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', MAIL_USERNAME)

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    ENV = 'production'
    PROTECT_PRODUCTION = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    if SQLALCHEMY_DATABASE_URI and SQLALCHEMY_DATABASE_URI.startswith('postgres://'):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace('postgres://', 'postgresql://', 1)

    # Enhanced production security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    PERMANENT_SESSION_LIFETIME = 1800  # 30 minutes
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Production-specific database settings
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 5,
        'pool_timeout': 30,
        'pool_recycle': 300,
        'pool_pre_ping': True,
        'connect_args': {
            'sslmode': 'require',
            'connect_timeout': 10,
            'application_name': 'financial_app_prod'
        }
    }

    @classmethod
    def init_app(cls, app):
        """Production-specific initialization"""
        Config.init_app(app)
        
        # Prevent modifications in production
        if cls.PROTECT_PRODUCTION:
            app.config['PREVENT_MODIFICATIONS'] = True
            app.config['CHARTS_OF_ACCOUNTS_PROTECTED'] = True


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    ENV = 'development'
    # Use separate database URL for development
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL', os.environ.get('DATABASE_URL'))
    if SQLALCHEMY_DATABASE_URI and SQLALCHEMY_DATABASE_URI.startswith('postgres://'):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace('postgres://', 'postgresql://', 1)

    # Protect production data in development
    PROTECT_PRODUCTION_DATA = True

    # Development-specific settings
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 2,
        'echo': True,
        'echo_pool': True,
        'pool_pre_ping': True,
        'connect_args': {
            'connect_timeout': 10
        }
    }

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('TEST_DATABASE_URL', 'sqlite:///test.db')

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}