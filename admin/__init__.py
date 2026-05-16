"""
Admin module for managing subscriptions and system-wide settings
This module is completely separate from core features to ensure their protection
"""
from flask import Blueprint, abort, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from functools import wraps

# Create admin blueprint with URL prefix
admin = Blueprint('admin', __name__, url_prefix='/admin')

def admin_required(f):
    """Decorator to protect admin routes"""
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('Access denied. Admin privileges required.', 'error')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


def maintenance_check(f):
    """Decorator to check if system is in maintenance mode"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        maintenance_mode = current_app.config.get('MAINTENANCE_MODE', False)
        if maintenance_mode and not current_user.is_admin:
            flash('System is currently in maintenance mode. Please try again later.', 'warning')
            return redirect(url_for('main.dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# Import routes after blueprint creation to avoid circular imports
from . import routes

# Protect all admin routes with admin_required decorator
@admin.before_request
def restrict_admin_access():
    """Ensure only admin users can access admin routes"""
    if not current_user.is_authenticated or not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('auth.login'))
