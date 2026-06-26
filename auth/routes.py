"""
Authentication routes including login, password reset functionality
"""
import logging
from flask import (
    render_template, redirect, url_for, flash,
    request, session, current_app
)
from flask_login import current_user, login_user, logout_user, login_required
from werkzeug.security import generate_password_hash

from . import auth
from models import db, User
from forms.auth import (
    LoginForm, RequestPasswordResetForm, ResetPasswordForm, 
    RegistrationForm
)

# Configure logging
logger = logging.getLogger(__name__)

@auth.route('/register', methods=['GET', 'POST'])
def register():
    """Handle new user registration"""
    try:
        if current_user.is_authenticated:
            logger.info(f"Authenticated user {current_user.id} redirected from registration")
            return redirect(url_for('main.dashboard'))

        form = RegistrationForm()
        if form.validate_on_submit():
            # Check if user already exists
            existing_user = User.query.filter_by(email=form.email.data.lower().strip()).first()

            if existing_user:
                flash('An account with this email already exists.', 'error')
                return redirect(url_for('auth.login'))

            try:
                # Create new user
                user = User(
                    username=form.username.data,
                    email=form.email.data.lower().strip()
                )
                user.set_password(form.password.data)
                db.session.add(user)
                db.session.commit()
                logger.info(f"New user registered successfully: {user.email}")
                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('auth.login'))

            except Exception as e:
                db.session.rollback()
                logger.error(f"Registration error: {str(e)}")
                flash('An error occurred during registration.', 'error')

        return render_template('auth/register.html', form=form)
    except Exception as e:
        logger.error(f"Unexpected error in registration: {str(e)}")
        flash('An unexpected error occurred.', 'error')
        return redirect(url_for('auth.login'))

@auth.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login with enhanced security and session management."""
    try:
        # Clear any existing session data
        session.clear()

        # If user is already authenticated, redirect appropriately
        if current_user.is_authenticated and not current_user.is_deleted:
            logger.info(f"Already authenticated user {current_user.id} redirected to dashboard")
            return redirect(url_for('main.dashboard'))

        form = LoginForm()
        if form.validate_on_submit():
            user = User.query.filter_by(email=form.email.data.lower().strip()).first()

            if not user:
                logger.warning(f"Login attempt with non-existent email: {form.email.data}")
                flash('Invalid email or password', 'error')
                return render_template('auth/login.html', form=form)

            if user.is_deleted:
                logger.warning(f"Login attempt by deleted user: {form.email.data}")
                flash('This account has been deleted. Please register again.', 'error')
                return render_template('auth/login.html', form=form)

            if not user.check_password(form.password.data):
                logger.warning(f"Failed login attempt for email: {form.email.data}")
                flash('Invalid email or password', 'error')
                return render_template('auth/login.html', form=form)

            # Login successful
            login_user(user, remember=form.remember_me.data)
            logger.info(f"User {user.email} logged in successfully")

            # Get the next page from the session or default to dashboard
            next_page = session.get('next', url_for('main.dashboard'))
            session.pop('next', None)  # Remove the next page from session

            flash('Login successful!', 'success')
            return redirect(next_page)

    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        flash('An error occurred during login.', 'error')

    # GET request or form validation failed
    return render_template('auth/login.html', form=form)

@auth.route('/logout')
@login_required
def logout():
    """Handle user logout with proper cleanup"""
    try:
        user_email = current_user.email
        logout_user()
        session.clear()  # Clear all session data
        logger.info(f"User {user_email} logged out successfully")
        flash('You have been logged out.', 'info')
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        flash('Error during logout.', 'error')
    return redirect(url_for('auth.login'))

@auth.route('/reset_password_request', methods=['GET', 'POST'])
def reset_password_request():
    """Handle password reset requests"""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))

    form = RequestPasswordResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data.lower().strip()).first()
        if user:
            flash('Check your email for instructions to reset your password', 'info')
            return redirect(url_for('auth.login'))
        else:
            flash('Email address not found', 'error')

    return render_template('auth/reset_password_request.html', form=form)

@auth.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """Handle password reset with token"""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))

    form = ResetPasswordForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data.lower().strip()).first()
        if user:
            user.set_password(form.password.data)
            db.session.commit()
            flash('Your password has been reset', 'success')
            return redirect(url_for('auth.login'))
        else:
            flash('Invalid email address', 'error')

    return render_template('auth/reset_password.html', form=form)