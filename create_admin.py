"""
Create (or reset) the admin user from environment variables.

No credentials are hardcoded. Set these before running:

    ADMIN_EMAIL=you@example.com ADMIN_PASSWORD=YOUR_STRONG_PASSWORD \
        [ADMIN_USERNAME=Admin] python create_admin.py
"""
import os
import sys
from datetime import datetime

from app import create_app
from models import db, User


def create_admin_user():
    """Create the admin user if it doesn't exist, else reset its password."""
    email = (os.environ.get('ADMIN_EMAIL') or '').lower().strip()
    password = os.environ.get('ADMIN_PASSWORD') or ''
    username = os.environ.get('ADMIN_USERNAME', 'Admin')

    if not email or not password:
        print("Set ADMIN_EMAIL and ADMIN_PASSWORD in the environment first.",
              file=sys.stderr)
        return 2

    app = create_app()
    if not app:
        print("Failed to create application (is DATABASE_URL set?).", file=sys.stderr)
        return 1

    with app.app_context():
        try:
            admin = User.query.filter_by(email=email).first()
            if not admin:
                admin = User(
                    username=username,
                    email=email,
                    is_admin=True,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    subscription_status='active',
                )
                admin.set_password(password)
                db.session.add(admin)
                print(f"Admin user created: {email}")
            else:
                admin.set_password(password)
                admin.is_admin = True
                admin.subscription_status = 'active'
                print(f"Admin user updated: {email}")
            db.session.commit()
            return 0
        except Exception as e:
            print(f"Error creating/updating admin user: {e}", file=sys.stderr)
            db.session.rollback()
            return 1


if __name__ == '__main__':
    sys.exit(create_admin_user())
