from app import create_app
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Module-level WSGI application object. Production servers such as gunicorn import
# this module and expect a callable named `app` (i.e. `gunicorn main:app`).
# Mirrors wsgi.py so both `main:app` and `wsgi:app` resolve to the same app.
app = create_app()


def main():
    try:
        logger.info("Starting application initialization...")

        if not app:
            raise ValueError("Application creation failed")
        logger.info("Application created successfully")

        # Get port from environment or default to 5000
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"Using port: {port}")

        # Start the Flask development server (only for `python main.py`)
        app.run(
            host='0.0.0.0',  # Allow external connections
            port=port,
            debug=True
        )

    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        logger.exception("Full stack trace:")
        sys.exit(1)

if __name__ == "__main__":
    main()