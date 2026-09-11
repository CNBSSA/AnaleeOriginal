"""
Reports module for financial reporting functionality
Handles generation and display of various financial reports including cashbook,
general ledger, and trial balance.
"""

import logging
import calendar
from datetime import datetime
from io import BytesIO
from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file, jsonify, current_app
from flask_login import login_required, current_user
from models import db, Transaction, Account, CompanySettings
from sqlalchemy import text, and_
from sqlalchemy.sql import func
from sqlalchemy.orm import contains_eager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the blueprint instance from __init__.py
from . import reports
from .trial_balance_service import (
    build_booksxperts_trial_balance_xlsx,
    build_trial_balance_payload,
    export_filename,
    load_trial_balance,
)
from .tb_share_tokens import create_share_token, verify_share_token, DEFAULT_MAX_AGE_SECONDS

def get_last_day_of_month(year: int, month: int) -> int:
    """
    Helper function to get the last day of a given month
    Args:
        year (int): The year
        month (int): The month (1-12)
    Returns:
        int: The last day of the month
    """
    return calendar.monthrange(year, month)[1]

@reports.route('/cashbook')
@login_required
def cashbook():
    """
    Display bank statement cashbook report with financial year and custom period filtering
    """
    try:
        company_settings = CompanySettings.query.filter_by(user_id=current_user.id).first()
        if not company_settings:
            flash('Please configure company settings first.')
            return redirect(url_for('main.company_settings'))

        # Get the earliest and latest transaction dates
        date_range = db.session.query(
            func.min(Transaction.date).label('min_date'),
            func.max(Transaction.date).label('max_date')
        ).filter(Transaction.user_id == current_user.id).first()

        # Set default dates if no transactions exist
        min_date = date_range.min_date or datetime.now()
        max_date = date_range.max_date or datetime.now()

        # Get available financial years based on data range
        financial_years = set()
        transactions = Transaction.query.filter_by(user_id=current_user.id).all()
        for t in transactions:
            if t.date.month > company_settings.financial_year_end:
                financial_years.add(t.date.year)
            else:
                financial_years.add(t.date.year - 1)
        financial_years = sorted(list(financial_years))

        # Default to current financial year if none selected
        if not financial_years:
            current_date = datetime.now()
            if current_date.month > company_settings.financial_year_end:
                financial_years = [current_date.year]
            else:
                financial_years = [current_date.year - 1]

        # Determine filtering mode and dates
        period_type = request.args.get('period_type', 'fy')
        selected_fy = None

        if period_type == 'custom':
            # Custom period filtering
            from_date = request.args.get('from_date')
            to_date = request.args.get('to_date')

            if from_date:
                from_date = datetime.strptime(from_date, '%Y-%m-%d').date()
            else:
                from_date = min_date

            if to_date:
                to_date = datetime.strptime(to_date, '%Y-%m-%d').date()
            else:
                to_date = max_date
        else:
            # Financial year filtering
            selected_fy = request.args.get('financial_year')
            if selected_fy:
                selected_fy = int(selected_fy)
            else:
                selected_fy = max(financial_years)

            # Calculate FY dates based on settings
            fy_end_month = company_settings.financial_year_end
            if fy_end_month == 12:
                from_date = datetime(selected_fy, 1, 1).date()
                to_date = datetime(selected_fy, 12, 31).date()
            else:
                from_date = datetime(selected_fy, fy_end_month + 1, 1).date()
                last_day = get_last_day_of_month(selected_fy + 1, fy_end_month)
                to_date = datetime(selected_fy + 1, fy_end_month, last_day).date()

        # Get transactions for the specified period
        transactions = Transaction.query.filter(
            Transaction.user_id == current_user.id,
            Transaction.date.between(from_date, to_date)
        ).order_by(Transaction.date).all()

        return render_template('reports/cashbook.html',
                             transactions=transactions,
                             start_date=from_date,
                             end_date=to_date,
                             min_date=min_date,
                             max_date=max_date,
                             financial_years=financial_years,
                             current_fy=selected_fy)

    except Exception as e:
        logger.error(f"Error generating cashbook report: {str(e)}")
        flash('Error generating cashbook report')
        return redirect(url_for('main.dashboard'))

@reports.route('/general-ledger')
@login_required
def general_ledger():
    """Display general ledger report"""
    try:
        company_settings = CompanySettings.query.filter_by(user_id=current_user.id).first()
        if not company_settings:
            flash('Please configure company settings first.')
            return redirect(url_for('main.company_settings'))
            
        accounts = Account.query.filter_by(user_id=current_user.id).order_by(Account.link).all()
        
        return render_template('reports/general_ledger.html',
                             accounts=accounts)
                             
    except Exception as e:
        logger.error(f"Error generating general ledger: {str(e)}")
        flash('Error generating general ledger')
        return redirect(url_for('main.dashboard'))

class BadPeriodError(ValueError):
    """A period selector on a trial-balance request could not be read."""


def _fy_probe_date(year: int, financial_year_end: int) -> datetime:
    """A date guaranteed to fall INSIDE the financial year starting in ``year``.

    QA register #11. ``CompanySettings.get_financial_year`` keeps ``start_year``
    one LESS than the calendar year it represents when the year-end is December
    (``start_date = datetime(start_year + 1, 1, 1)``). Its date-derived path
    compensates for that; the ``year=`` keyword does not, so asking for
    ``financial_year=2025`` on a calendar-year client returned calendar 2026.

    Rather than change that long-standing convention — every other Analee report
    depends on it, and it lives in a frozen module — this converts the request
    into a probe date and lets the already-correct date-derived path resolve it.
    """
    if financial_year_end == 12:
        return datetime(year, 6, 30)          # mid-calendar-year
    return datetime(year, financial_year_end + 1, 15)   # just after the year-end


def _requested_period(user_id: int | None = None) -> dict:
    """Optional period selectors shared by the trial-balance page, the export, the
    JSON API and the share endpoint, passed through to ``load_trial_balance``:

    - ``as_at=YYYY-MM-DD`` — any date inside the wanted financial year. A
      consumer that knows its own year-end (THE ACCOUNTANTS) passes exactly
      that date and receives that year's balance.
    - ``financial_year=YYYY`` — the year the financial year STARTS in. For a
      December year-end that is simply the calendar year.

    Both are resolved to an ``as_at`` date, so one code path decides the year
    (#11). ``financial_year`` still wins when both are supplied, as before.

    Nothing given → the client's current financial year, exactly as before.
    """
    period: dict = {}
    raw_as_at = (request.args.get('as_at') or '').strip()
    raw_year = (request.args.get('financial_year') or '').strip()
    if raw_as_at:
        try:
            period['as_at'] = datetime.strptime(raw_as_at, '%Y-%m-%d')
        except ValueError:
            raise BadPeriodError('as_at must be a date in the form YYYY-MM-DD.')
    if raw_year:
        try:
            year = int(raw_year)
        except ValueError:
            raise BadPeriodError('financial_year must be a four-digit year.')
        if not 1900 <= year <= 2999:
            raise BadPeriodError('financial_year must be a four-digit year.')
        settings = CompanySettings.query.filter_by(user_id=user_id).first() \
            if user_id is not None else None
        if settings is None:
            # No settings to read the year-end from: keep the old keyword so the
            # caller still gets a period rather than a silently wrong one, and let
            # load_trial_balance raise its own "configure company settings" error.
            period.pop('as_at', None)
            period['year'] = year
        else:
            period['as_at'] = _fy_probe_date(
                year, int(settings.financial_year_end or 12))
    return period


def _period_query_string() -> dict:
    """The period selectors as the page received them, for re-attaching to the
    page's own actions (#10). The Excel, JSON and share-link controls built their
    URLs with no query string, so acting on a closed year silently served the
    current one."""
    out = {}
    for key in ('as_at', 'financial_year'):
        value = (request.args.get(key) or '').strip()
        if value:
            out[key] = value
    return out


def _period_options(user_id: int, selected_end: datetime, *, back: int = 5) -> list[dict]:
    """Years offered by the page's period selector (#10).

    The capability to ask for a closed year landed on 2026-09-07 but was reachable
    only by hand-editing the URL, which is not a path an accountant should need.
    Labels come from ``get_financial_year`` itself so they can never describe a
    range the report would not actually return.
    """
    settings = CompanySettings.query.filter_by(user_id=user_id).first()
    if settings is None:
        return []
    fy_end = int(settings.financial_year_end or 12)
    current = settings.get_financial_year()
    current_start_year = current['start_date'].year if fy_end == 12 \
        else current['start_date'].year
    options = []
    for year in range(current_start_year, current_start_year - back - 1, -1):
        dates = settings.get_financial_year(date=_fy_probe_date(year, fy_end))
        options.append({
            'value': year,
            'label': (f"{dates['start_date'].strftime('%d %b %Y')} – "
                      f"{dates['end_date'].strftime('%d %b %Y')}"),
            'selected': dates['end_date'].date() == selected_end.date(),
        })
    return options


@reports.route('/trial-balance')
@login_required
def trial_balance():
    """Display trial balance report"""
    try:
        ctx = load_trial_balance(current_user.id, **_requested_period(current_user.id))
        # QA register #9: the template used to re-derive each row by summing
        # `account.transactions` — the whole ORM relationship, with no period bound
        # — while the totals came from the service. Once the service was correctly
        # bounded to the period end (2026-09-07) the page contradicted itself.
        # `ctx.rows` is the service's own answer, keyed by link, which is unique per
        # user (ix_account_user_link). Accounts absent from it have a zero balance
        # and still render as 0.00, so the row set is unchanged.
        amounts_by_link = {row.link: row.amount for row in ctx.rows}
        return render_template(
            'reports/trial_balance.html',
            accounts=ctx.accounts,
            amounts_by_link=amounts_by_link,
            start_date=ctx.start_date,
            end_date=ctx.end_date,
            total_debits=ctx.total_debits,
            total_credits=ctx.total_credits,
            tb_balanced=ctx.total_debits == ctx.total_credits,
            period_options=_period_options(current_user.id, ctx.end_date),
            period_query=_period_query_string(),
        )
    except BadPeriodError as exc:
        flash(str(exc))
        return redirect(url_for('reports.trial_balance'))
    except ValueError:
        flash('Please configure company settings first.')
        return redirect(url_for('main.company_settings'))
    except Exception as e:
        logger.error(f"Error generating trial balance: {str(e)}, Stack trace: {str(e.__traceback__)}")
        flash('Error loading transaction data. Please try again.')
        return redirect(url_for('main.dashboard'))


@reports.route('/trial-balance/export')
@login_required
def trial_balance_export():
    """Download trial balance as BooksXperts-compatible Excel."""
    try:
        company_settings = CompanySettings.query.filter_by(user_id=current_user.id).first()
        if not company_settings:
            flash('Please configure company settings first.')
            return redirect(url_for('main.company_settings'))

        ctx = load_trial_balance(current_user.id, **_requested_period(current_user.id))
        if not ctx.rows:
            flash('No trial balance amounts to export for this period.')
            return redirect(url_for('reports.trial_balance'))

        xlsx = build_booksxperts_trial_balance_xlsx(
            ctx.rows,
            company_name=company_settings.company_name,
            period_end=ctx.end_date,
        )
        return send_file(
            BytesIO(xlsx),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=export_filename(ctx.end_date),
        )
    except BadPeriodError as exc:
        flash(str(exc))
        return redirect(url_for('reports.trial_balance'))
    except Exception as e:
        logger.error(f"Error exporting trial balance: {str(e)}, Stack trace: {str(e.__traceback__)}")
        flash('Could not export trial balance. Please try again.')
        return redirect(url_for('reports.trial_balance'))


def _trial_balance_payload_for_user(user_id: int, **period) -> tuple[dict, CompanySettings]:
    company_settings = CompanySettings.query.filter_by(user_id=user_id).first()
    if not company_settings:
        raise ValueError('Company settings are not configured.')
    ctx = load_trial_balance(user_id, **period)
    if not ctx.rows:
        raise ValueError('No trial balance amounts for this period.')
    payload = build_trial_balance_payload(
        ctx,
        user_id=user_id,
        company_name=company_settings.company_name,
        registration_number=company_settings.registration_number,
    )
    return payload, company_settings


@reports.route('/api/trial-balance')
@login_required
def trial_balance_api():
    """Authenticated JSON trial balance for downstream import (BooksXperts / Accountants)."""
    try:
        payload, _ = _trial_balance_payload_for_user(
            current_user.id, **_requested_period(current_user.id))
        return jsonify(payload)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as e:
        logger.error(f"Error serving trial balance API: {str(e)}")
        return jsonify({'error': 'Could not load trial balance.'}), 500


@reports.route('/api/trial-balance/share')
@login_required
def trial_balance_share_link():
    """Create a time-limited signed URL for trial balance JSON (24h)."""
    try:
        period = _requested_period(current_user.id)
        _trial_balance_payload_for_user(current_user.id, **period)
        token = create_share_token(current_user.id, secret_key=current_app.config['SECRET_KEY'])
        # #10: the token names only the company, so without this the copied link
        # always meant "current financial year" however the page was filtered.
        share_kwargs = {}
        if 'as_at' in period:
            share_kwargs['as_at'] = period['as_at'].strftime('%Y-%m-%d')
        elif 'year' in period:
            share_kwargs['financial_year'] = period['year']
        share_url = url_for('reports.trial_balance_shared', token=token,
                            _external=True, **share_kwargs)
        return jsonify({
            'share_url': share_url,
            'expires_in_seconds': DEFAULT_MAX_AGE_SECONDS,
            'format': 'application/json',
        })
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as e:
        logger.error(f"Error creating trial balance share link: {str(e)}")
        return jsonify({'error': 'Could not create share link.'}), 500


@reports.route('/api/trial-balance/shared/<token>')
def trial_balance_shared(token):
    """Fetch trial balance JSON via signed share token (no login required)."""
    from itsdangerous import BadSignature, SignatureExpired

    try:
        user_id = verify_share_token(token, secret_key=current_app.config['SECRET_KEY'])
        # The token names the company; the consumer names the year (or gets the
        # current one). A BadPeriodError is a ValueError → 400 below.
        payload, _ = _trial_balance_payload_for_user(user_id, **_requested_period(user_id))
        response = jsonify(payload)
        response.headers['Cache-Control'] = 'no-store'
        return response
    except SignatureExpired:
        return jsonify({'error': 'Share link has expired. Generate a new link from Analee.'}), 410
    except BadSignature:
        return jsonify({'error': 'Invalid share link.'}), 403
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as e:
        logger.error(f"Error serving shared trial balance: {str(e)}")
        return jsonify({'error': 'Could not load trial balance.'}), 500


@reports.route('/financial-position')
@login_required
def financial_position():
    """Display statement of financial position"""
    try:
        company_settings = CompanySettings.query.filter_by(user_id=current_user.id).first()
        if not company_settings:
            flash('Please configure company settings first.')
            return redirect(url_for('main.company_settings'))

        # Get the earliest and latest transaction dates
        date_range = db.session.query(
            func.min(Transaction.date).label('min_date'),
            func.max(Transaction.date).label('max_date')
        ).filter(Transaction.user_id == current_user.id).first()

        # Set default dates if no transactions exist
        min_date = date_range.min_date or datetime.now()
        max_date = date_range.max_date or datetime.now()

        # Get available financial years based on data range
        financial_years = set()
        transactions = Transaction.query.filter_by(user_id=current_user.id).all()
        for t in transactions:
            if t.date.month > company_settings.financial_year_end:
                financial_years.add(t.date.year)
            else:
                financial_years.add(t.date.year - 1)
        financial_years = sorted(list(financial_years))

        # Default to current financial year if none selected
        if not financial_years:
            current_date = datetime.now()
            if current_date.month > company_settings.financial_year_end:
                financial_years = [current_date.year]
            else:
                financial_years = [current_date.year - 1]

        # Determine filtering mode and dates
        period_type = request.args.get('period_type', 'fy')
        selected_fy = None

        if period_type == 'custom':
            # Custom period filtering
            from_date = request.args.get('from_date')
            to_date = request.args.get('to_date')

            if from_date:
                from_date = datetime.strptime(from_date, '%Y-%m-%d').date()
            else:
                from_date = min_date

            if to_date:
                to_date = datetime.strptime(to_date, '%Y-%m-%d').date()
            else:
                to_date = max_date
        else:
            # Financial year filtering
            selected_fy = request.args.get('financial_year')
            if selected_fy:
                selected_fy = int(selected_fy)
            else:
                selected_fy = max(financial_years)

            # Calculate FY dates based on settings
            fy_end_month = company_settings.financial_year_end
            if fy_end_month == 12:
                from_date = datetime(selected_fy, 1, 1).date()
                to_date = datetime(selected_fy, 12, 31).date()
            else:
                from_date = datetime(selected_fy, fy_end_month + 1, 1).date()
                last_day = get_last_day_of_month(selected_fy + 1, fy_end_month)
                to_date = datetime(selected_fy + 1, fy_end_month, last_day).date()

        # Get accounts with their transactions for the period
        accounts = Account.query.filter_by(user_id=current_user.id).all()

        # Initialize asset and liability accounts with balances
        asset_accounts = []
        liability_accounts = []
        total_assets = 0
        total_liabilities = 0

        for account in accounts:
            # Calculate account balance for the period
            balance = db.session.query(func.sum(Transaction.amount)).filter(
                Transaction.account_id == account.id,
                Transaction.date <= to_date
            ).scalar() or 0

            account_data = {
                'name': account.name,
                'balance': abs(balance)  # Always show positive numbers in report
            }

            # Categorize accounts
            if account.category in ['Asset', 'Current Asset', 'Fixed Asset']:
                asset_accounts.append(account_data)
                total_assets += balance if balance > 0 else 0
            elif account.category in ['Liability', 'Current Liability', 'Long Term Liability']:
                liability_accounts.append(account_data)
                total_liabilities += abs(balance) if balance < 0 else 0

        return render_template('reports/financial_position.html',
                             start_date=from_date,
                             end_date=to_date,
                             min_date=min_date,
                             max_date=max_date,
                             financial_years=financial_years,
                             current_fy=selected_fy,
                             asset_accounts=asset_accounts,
                             liability_accounts=liability_accounts,
                             total_assets=total_assets,
                             total_liabilities=total_liabilities)

    except Exception as e:
        logger.error(f"Error generating financial position: {str(e)}")
        flash('Error generating financial position statement')
        return redirect(url_for('main.dashboard'))

@reports.route('/income-statement')
@login_required
def income_statement():
    """Display income statement"""
    try:
        company_settings = CompanySettings.query.filter_by(user_id=current_user.id).first()
        if not company_settings:
            flash('Please configure company settings first.')
            return redirect(url_for('main.company_settings'))

        # Get the earliest and latest transaction dates
        date_range = db.session.query(
            func.min(Transaction.date).label('min_date'),
            func.max(Transaction.date).label('max_date')
        ).filter(Transaction.user_id == current_user.id).first()

        # Set default dates if no transactions exist
        min_date = date_range.min_date or datetime.now()
        max_date = date_range.max_date or datetime.now()

        # Get available financial years based on data range
        financial_years = set()
        transactions = Transaction.query.filter_by(user_id=current_user.id).all()
        for t in transactions:
            if t.date.month > company_settings.financial_year_end:
                financial_years.add(t.date.year)
            else:
                financial_years.add(t.date.year - 1)
        financial_years = sorted(list(financial_years))

        # Default to current financial year if none selected
        if not financial_years:
            current_date = datetime.now()
            if current_date.month > company_settings.financial_year_end:
                financial_years = [current_date.year]
            else:
                financial_years = [current_date.year - 1]

        # Determine filtering mode and dates
        period_type = request.args.get('period_type', 'fy')
        selected_fy = None

        if period_type == 'custom':
            # Custom period filtering
            from_date = request.args.get('from_date')
            to_date = request.args.get('to_date')

            if from_date:
                from_date = datetime.strptime(from_date, '%Y-%m-%d').date()
            else:
                from_date = min_date

            if to_date:
                to_date = datetime.strptime(to_date, '%Y-%m-%d').date()
            else:
                to_date = max_date
        else:
            # Financial year filtering
            selected_fy = request.args.get('financial_year')
            if selected_fy:
                selected_fy = int(selected_fy)
            else:
                selected_fy = max(financial_years)

            # Calculate FY dates based on settings
            fy_end_month = company_settings.financial_year_end
            if fy_end_month == 12:
                from_date = datetime(selected_fy, 1, 1).date()
                to_date = datetime(selected_fy, 12, 31).date()
            else:
                from_date = datetime(selected_fy, fy_end_month + 1, 1).date()
                last_day = get_last_day_of_month(selected_fy + 1, fy_end_month)
                to_date = datetime(selected_fy + 1, fy_end_month, last_day).date()

        # Get accounts with their transactions for the period
        accounts = Account.query.filter_by(user_id=current_user.id).all()

        # Initialize income and expense accounts with balances
        income_accounts = []
        expense_accounts = []
        total_income = 0
        total_expenses = 0

        for account in accounts:
            # Calculate account balance for the period
            balance = db.session.query(func.sum(Transaction.amount)).filter(
                Transaction.account_id == account.id,
                Transaction.date.between(from_date, to_date)
            ).scalar() or 0

            account_data = {
                'name': account.name,
                'balance': abs(balance)  # Always show positive numbers in report
            }

            # Categorize accounts
            if account.category in ['Income', 'Revenue']:
                income_accounts.append(account_data)
                total_income += balance if balance > 0 else 0
            elif account.category in ['Expense', 'Cost of Sales']:
                expense_accounts.append(account_data)
                total_expenses += abs(balance) if balance < 0 else 0

        return render_template('reports/income_statement.html',
                            start_date=from_date,
                            end_date=to_date,
                            min_date=min_date,
                            max_date=max_date,
                            financial_years=financial_years,
                            current_fy=selected_fy,
                            income_accounts=income_accounts,
                            expense_accounts=expense_accounts,
                            total_income=total_income,
                            total_expenses=total_expenses)

    except Exception as e:
        logger.error(f"Error generating income statement: {str(e)}")
        flash('Error generating income statement')
        return redirect(url_for('main.dashboard'))
