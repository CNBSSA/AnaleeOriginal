"""
Authentication routes including login, password reset functionality
"""
import logging
import os
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
from security import login_throttle
from password_reset_tokens import (
    BadSignature,
    SignatureExpired,
    create_password_reset_token,
    token_matches_current_password,
    verify_password_reset_token,
)

# Configure logging
logger = logging.getLogger(__name__)


def _email_ref(email: str) -> str:
    """A short, stable reference for an e-mail address in a log line — never
    the address itself (work-orders #69, R6). Railway logs are read by several
    people and retained; support can still correlate repeated attempts by the
    same reference."""
    import hashlib
    return 'email#' + hashlib.sha256((email or '').strip().lower().encode('utf-8')).hexdigest()[:12]

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

            if existing_user and not existing_user.is_deleted:
                flash('An account with this email already exists.', 'error')
                return redirect(url_for('auth.login'))

            if existing_user:
                # A previously soft-deleted account holds this address. Move its
                # login identifiers aside so the person can sign up again — the
                # old row and all of its accounting records stay exactly where
                # they are. Deleting it would destroy books the Companies Act
                # and SARS require be kept (docs/DATA_RETENTION.md).
                logger.info(
                    'Releasing identifiers from soft-deleted user_id=%s so %s '
                    'can be re-registered; records retained',
                    existing_user.id, _email_ref(existing_user.email))
                existing_user.release_identifiers_for_reregistration()
                db.session.flush()

            try:
                # Create new user
                user = User(
                    username=form.username.data,
                    email=form.email.data.lower().strip()
                )
                user.set_password(form.password.data)
                db.session.add(user)
                db.session.commit()
                logger.info("New user registered successfully: user_id=%s", user.id)
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
        # Clear any existing session data (session-fixation hygiene on the way
        # in). Preserve the flash queue across it, though: everything that
        # redirects HERE to tell the user something — "your password has been
        # reset", "check your email" — had its message destroyed by this line
        # before the page rendered, so the user saw silence. The clear keeps
        # doing its job; the messages now survive it. (QA audit, 2026-09-04.)
        _pending_flashes = session.get('_flashes')
        session.clear()
        if _pending_flashes:
            session['_flashes'] = _pending_flashes

        # If user is already authenticated, redirect appropriately
        if current_user.is_authenticated and not current_user.is_deleted:
            logger.info(f"Already authenticated user {current_user.id} redirected to dashboard")
            return redirect(url_for('main.dashboard'))

        form = LoginForm()
        if form.validate_on_submit():
            submitted_email = form.email.data.lower().strip()

            # Abuse protection (work-orders #69, R1): too many failures for this
            # address or from this client in the last 15 minutes → refuse before
            # any password is checked. Database-backed, hashed keys, fail-open —
            # see security/login_throttle.py.
            if login_throttle.locked(submitted_email):
                flash(login_throttle.LOCKED_MESSAGE, 'error')
                response = current_app.make_response(
                    render_template('auth/login.html', form=form))
                response.status_code = 429
                response.headers['Retry-After'] = str(login_throttle.retry_after_seconds())
                return response

            user = User.query.filter_by(email=submitted_email).first()

            if not user:
                logger.warning("Login attempt with non-existent email: %s",
                               _email_ref(submitted_email))
                login_throttle.record_failure(submitted_email)
                flash('Invalid email or password', 'error')
                return render_template('auth/login.html', form=form)

            if user.is_deleted:
                # Same answer as an unknown address (work-orders #69, R4): the
                # old wording told any visitor who HAD been a customer. The
                # register path explains re-registration to the right person.
                logger.warning("Login attempt by deleted user_id=%s", user.id)
                login_throttle.record_failure(submitted_email)
                flash('Invalid email or password', 'error')
                return render_template('auth/login.html', form=form)

            if not user.check_password(form.password.data):
                logger.warning("Failed login attempt for user_id=%s", user.id)
                login_throttle.record_failure(submitted_email)
                flash('Invalid email or password', 'error')
                return render_template('auth/login.html', form=form)

            # Login successful
            login_throttle.record_success(submitted_email)
            login_user(user, remember=form.remember_me.data)
            logger.info("User user_id=%s logged in successfully", user.id)

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
        user_id = current_user.id
        logout_user()
        session.clear()  # Clear all session data
        logger.info("User user_id=%s logged out successfully", user_id)
        flash('You have been logged out.', 'info')
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        flash('Error during logout.', 'error')
    return redirect(url_for('auth.login'))

# Same answer whether or not the address is registered. The old wording
# ("Email address not found") turned this page into a free membership
# checker for anyone who wanted to know who our customers are.
_RESET_REQUESTED_MESSAGE = (
    'If that email address has an account, a reset link has been created for '
    'it. The link is valid for one hour. If nothing arrives, contact support '
    'and we will reset it for you.'
)

_RESET_LINK_DEAD_MESSAGE = (
    'That password reset link is no longer valid — it has expired, or it has '
    'already been used. Request a new one below.'
)


@auth.route('/reset_password_request', methods=['GET', 'POST'])
def reset_password_request():
    """Handle password reset requests"""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))

    form = RequestPasswordResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(
            email=form.email.data.lower().strip()).first()
        if user:
            token = create_password_reset_token(
                user, secret_key=current_app.config['SECRET_KEY'])
            reset_url = url_for('auth.reset_password', token=token,
                                _external=True)
            # Email delivery is not wired up in this app yet, so nothing is
            # actually sent. Saying "check your email" while sending nothing is
            # how a locked-out customer ends up stranded, so we do not say it.
            # Festus has no shell but does have Railway logs, so the link can be
            # surfaced there to unstick someone — deliberately OFF by default,
            # because a reset link in a log file is an account-takeover token in
            # a log file. Turn it on only while helping a specific customer.
            if (os.environ.get('ANALEE_PASSWORD_RESET_LOG_LINK') or '').strip() == '1':
                logger.warning('PASSWORD RESET LINK for %s: %s',
                               user.email, reset_url)
            else:
                logger.info('Password reset requested for user_id=%s '
                            '(link not logged; set '
                            'ANALEE_PASSWORD_RESET_LOG_LINK=1 to surface it)',
                            user.id)
        else:
            logger.info('Password reset requested for an unknown address')
        flash(_RESET_REQUESTED_MESSAGE, 'info')
        return redirect(url_for('auth.login'))

    return render_template('auth/reset_password_request.html', form=form)


@auth.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """Handle password reset with token.

    The TOKEN decides whose password changes — never anything posted in the
    form. Do not reintroduce an email (or user id) field here: combined with a
    lookup, that is unauthenticated account takeover. See
    password_reset_tokens.py and tests/test_password_reset.py.
    """
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))

    secret_key = current_app.config['SECRET_KEY']

    def _dead_link():
        flash(_RESET_LINK_DEAD_MESSAGE, 'error')
        return redirect(url_for('auth.reset_password_request'))

    # Verified on GET as well as POST, so an invalid link never renders a form
    # that looks like it will work.
    try:
        user_id = verify_password_reset_token(token, secret_key=secret_key)
    except (BadSignature, SignatureExpired):
        return _dead_link()

    user = User.query.get(user_id)
    if user is None or user.is_deleted:
        return _dead_link()

    # Bound to the password hash at issue time: once this token has been used
    # (or the password changed by any other route) it stops working.
    try:
        if not token_matches_current_password(token, user, secret_key=secret_key):
            return _dead_link()
    except (BadSignature, SignatureExpired):
        return _dead_link()

    form = ResetPasswordForm()
    if form.validate_on_submit():
        user.set_password(form.password.data)
        db.session.commit()
        logger.info('Password reset completed for user_id=%s', user.id)
        flash('Your password has been reset. You can log in now.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('auth/reset_password.html', form=form)