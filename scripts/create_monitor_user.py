#!/usr/bin/env python3
"""
Provision the synthetic-monitor user (idempotent).

Creates — or resets the password of — a low-privilege user from MONITOR_EMAIL /
MONITOR_PASSWORD, and ensures it has company settings so the monitored report
renders. Safe to run repeatedly. Run once on the deployment (e.g. Railway shell),
using the SAME credentials you put in the monitor's GitHub secrets:

    MONITOR_EMAIL=monitor@yourco.com MONITOR_PASSWORD='...' \
        python scripts/create_monitor_user.py

Requires DATABASE_URL (and the app's normal env) to be set.
"""
import os
import sys
from datetime import datetime

from app import create_app
from models import db, User, CompanySettings

USERNAME = "synthetic-monitor"
COMPANY_NAME = "Synthetic Monitor"
FINANCIAL_YEAR_END = 12  # December -> calendar-year financial year


def main():
    email = (os.environ.get("MONITOR_EMAIL") or "").lower().strip()
    password = os.environ.get("MONITOR_PASSWORD") or ""
    if not email or not password:
        print("Set MONITOR_EMAIL and MONITOR_PASSWORD in the environment first.",
              file=sys.stderr)
        return 2

    app = create_app()
    if not app:
        print("Failed to create application (is DATABASE_URL set?).", file=sys.stderr)
        return 1

    with app.app_context():
        try:
            user = User.query.filter_by(email=email).first()
            if user is None:
                user = User(
                    username=USERNAME,
                    email=email,
                    is_admin=False,
                    subscription_status="active",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                user.set_password(password)
                db.session.add(user)
                db.session.flush()  # assign user.id
                print(f"Created monitoring user: {email}")
            else:
                user.set_password(password)
                user.is_deleted = False
                user.subscription_status = "active"
                print(f"Updated monitoring user: {email}")

            settings = CompanySettings.query.filter_by(user_id=user.id).first()
            if settings is None:
                db.session.add(CompanySettings(
                    user_id=user.id,
                    company_name=COMPANY_NAME,
                    financial_year_end=FINANCIAL_YEAR_END,
                ))
                print("Added company settings for the monitoring user.")

            db.session.commit()
            print("Done. The synthetic monitor can now log in as this user.")
            return 0
        except Exception as exc:
            db.session.rollback()
            print(f"Error provisioning monitoring user: {exc}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())
