"""Bank-statement OCR routes: upload a PDF statement, review extracted rows,
confirm into transactions. (Analee is strictly cash-basis; receipt OCR removed.)"""
import logging
from datetime import datetime, timedelta

from flask import render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user

from models import db, UploadedFile, Account, Transaction
from . import ocr
from .service import ALLOWED_DOCUMENT_TYPES, mark_duplicates
from .statement_extractor import extract_bank_statement, MAX_PDF_BYTES

logger = logging.getLogger(__name__)


def _user_accounts():
    """The current user's active accounts. Never raises: on any DB error it
    returns [] (and rolls back) so the upload page still renders instead of
    returning a 500 to the user."""
    try:
        return (Account.query
                .filter_by(user_id=current_user.id, is_active=True)
                .order_by(Account.name)
                .all())
    except Exception as exc:
        logger.error(f"Could not load accounts for the OCR upload page: {exc}")
        try:
            db.session.rollback()
        except Exception:
            pass
        return []


def _start_autoprocess(file_id, user_id) -> bool:
    """Kick off post-import categorisation. Never fails the import."""
    try:
        from flask import current_app
        from services.auto_process import schedule_file_autoprocess
        return schedule_file_autoprocess(
            current_app._get_current_object(), file_id, user_id)
    except Exception:
        logger.exception("Could not schedule auto-process for file %s", file_id)
        return False


def _flag_duplicate_rows(rows):
    """Flag extracted rows that likely match the current user's existing
    transactions, so the review screen can pre-exclude them. Best-effort: any
    failure leaves rows unflagged rather than blocking the import."""
    try:
        date_strings = {r['date'] for r in rows if r.get('date')}
        if not date_strings:
            for r in rows:
                r['duplicate'] = False
            return rows
        date_objs = [datetime.strptime(d, '%Y-%m-%d') for d in date_strings]
        existing_q = Transaction.query.filter(
            Transaction.user_id == current_user.id,
            Transaction.date >= min(date_objs),
            Transaction.date < max(date_objs) + timedelta(days=1),
        ).all()
        existing = [
            (t.date.strftime('%Y-%m-%d'), t.amount, (t.description or ''))
            for t in existing_q
        ]
        return mark_duplicates(rows, existing)
    except Exception as e:
        logger.error(f"Duplicate flagging skipped: {str(e)}")
        for r in rows:
            r.setdefault('duplicate', False)
        return rows


@ocr.route('/statement', methods=['GET', 'POST'])
@login_required
def upload_statement():
    """Phase 2: upload a PDF bank statement; on success show the review screen."""
    accounts = _user_accounts()

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not file.filename:
            flash('Please choose a PDF bank statement to upload.', 'error')
            return redirect(url_for('ocr.upload_statement'))

        ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
        if ext not in ALLOWED_DOCUMENT_TYPES:
            flash('Unsupported file type. Please upload a PDF.', 'error')
            return redirect(url_for('ocr.upload_statement'))

        pdf_bytes = file.read()
        return review_extracted_statement(
            pdf_bytes, file.filename, request.form.get('account_id', ''),
            request.form.get('opening_balance'), request.form.get('closing_balance'),
            accounts=accounts)

    return render_template('ocr/statement_upload.html', accounts=accounts)


def review_extracted_statement(pdf_bytes, filename, account_id, opening_balance,
                               closing_balance, *, accounts=None, back_to='ocr.upload_statement'):
    """Extract a PDF statement and show the review screen — the body of the
    upload route, shared with the FileIt tunnel (2026-09-29) so a fetched PDF
    takes exactly the path an uploaded one does; ``/ocr/statement/confirm``
    then finishes the import unchanged."""
    if accounts is None:
        accounts = _user_accounts()
    if not pdf_bytes:
        flash('The uploaded file is empty.', 'error')
        return redirect(url_for(back_to))
    if len(pdf_bytes) > MAX_PDF_BYTES:
        flash('PDF is too large (max 32 MB).', 'error')
        return redirect(url_for(back_to))

    outcome = extract_bank_statement(
        pdf_bytes,
        opening_balance=opening_balance,
        closing_balance=closing_balance,
    )
    if not outcome.ok:
        flash(outcome.error or 'Could not read any transactions from that PDF.', 'error')
        return redirect(url_for(back_to))

    rows = _flag_duplicate_rows(outcome.rows)
    return render_template(
        'ocr/review.html',
        rows=rows,
        accounts=accounts,
        account_id=account_id or '',
        filename=filename,
        statement_header=outcome.header,
        report_card=outcome.report_card,
        extraction_method=outcome.method,
    )


@ocr.route('/statement/confirm', methods=['POST'])
@login_required
def confirm_receipt():
    """Persist the user-reviewed statement rows as transactions.

    (Endpoint name kept as ``confirm_receipt`` for URL/template stability; it is
    the shared review→import save path for bank-statement OCR.)"""
    dates = request.form.getlist('date')
    descriptions = request.form.getlist('description')
    amounts = request.form.getlist('amount')
    account_id = request.form.get('account_id') or None
    filename = request.form.get('filename') or 'statement'

    # Per-row include filter (Phase 2.1). The review screen unchecks likely
    # duplicates by default; only checked rows carry an 'include' value equal to
    # their row index. The hidden 'has_include_filter' field disambiguates
    # "filter present, nothing checked" from "no filter at all" (import everything).
    # Decision 2026-09-30: a statement whose balances were read and do not add
    # up imports only after the user ticks "I've checked the flagged rows" on
    # the review screen (the box is required there; this is the server half).
    if request.form.get('needs_ack') and not request.form.get('ack_unreconciled'):
        flash("Nothing was imported. This statement doesn't add up — tick "
              "\"I've checked the flagged rows\" to import it anyway, or fix the "
              "rows first.", 'error')
        return redirect(url_for('ocr.upload_statement'))

    has_include_filter = bool(request.form.get('has_include_filter'))
    included_indexes = set(request.form.getlist('include'))

    # Resolve the (optional) target account, scoped to the current user.
    account = None
    if account_id:
        try:
            account = Account.query.filter_by(
                id=int(account_id), user_id=current_user.id).first()
        except (TypeError, ValueError):
            account = None

    parsed_rows = []
    unreadable_dates = []
    for index, (raw_date, raw_desc, raw_amount) in enumerate(zip(dates, descriptions, amounts)):
        if has_include_filter and str(index) not in included_indexes:
            continue
        description = (raw_desc or '').strip()
        if not description:
            continue
        try:
            amount = float(str(raw_amount).replace(',', '').replace('$', '').strip())
        except (TypeError, ValueError):
            continue
        # A date we cannot read is NEVER silently replaced with today's date.
        # It used to be, and that is a bookkeeping defect, not a convenience:
        # the transaction lands in whatever VAT/tax period the import happened
        # to run in, the books look complete, and nothing anywhere says the
        # date was invented. Refuse the row and name it instead — the review
        # screen the user just came from is editable, so this is a correction
        # they can actually make. (QA audit, 2026-09-04.)
        try:
            date_value = datetime.strptime((raw_date or '').strip(), '%Y-%m-%d')
        except (TypeError, ValueError):
            unreadable_dates.append(index + 1)
            continue
        parsed_rows.append((date_value, description, amount))

    def _rows_phrase(rows):
        listed = ', '.join(str(r) for r in rows[:10])
        return listed + ('…' if len(rows) > 10 else '')

    if not parsed_rows:
        if unreadable_dates:
            flash(
                'Nothing was imported. The date could not be read on '
                f'{len(unreadable_dates)} row(s): {_rows_phrase(unreadable_dates)}. '
                'Upload the PDF again and, on the review screen, type each of those '
                'dates as YYYY-MM-DD (for example 2026-03-15) before importing.',
                'error',
            )
        else:
            flash('No valid rows to import. Please review the extracted values.', 'error')
        return redirect(url_for('ocr.upload_statement'))

    try:
        uploaded_file = UploadedFile(
            filename=filename,
            user_id=current_user.id,
            upload_date=datetime.utcnow(),
            bank_account_id=account.id if account else None,
        )
        db.session.add(uploaded_file)
        db.session.flush()  # get uploaded_file.id without a second round-trip

        for date_value, description, amount in parsed_rows:
            db.session.add(Transaction(
                date=date_value,
                description=description,
                amount=amount,
                file_id=uploaded_file.id,
                user_id=current_user.id,
                account_id=account.id if account else None,
            ))

        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error importing statement transactions: {str(e)}")
        flash('Could not import the transactions. Please try again.', 'error')
        return redirect(url_for('ocr.upload_statement'))

    # Start categorising/explaining immediately, off-request. The upload
    # returns now; the work continues on a background thread (see
    # services/auto_process.py for why it must not run inside the request).
    started = _start_autoprocess(uploaded_file.id, current_user.id)

    if unreadable_dates:
        # Never report a partial import as a clean one.
        flash(
            f'Imported {len(parsed_rows)} transaction(s). '
            f'{len(unreadable_dates)} row(s) were NOT imported because the date '
            f'could not be read: {_rows_phrase(unreadable_dates)}. '
            'To add them, upload the PDF again and type those dates as YYYY-MM-DD '
            'on the review screen.',
            'warning',
        )
    else:
        flash(f'Imported {len(parsed_rows)} transaction(s).', 'success')
    if started:
        flash('Analee is categorising and explaining them now — refresh this '
              'page in a minute to see what needs your eye.', 'info')
    # Land on the statement just imported, not the empty CSV upload form.
    return redirect(url_for('main.analyze', file_id=uploaded_file.id))
