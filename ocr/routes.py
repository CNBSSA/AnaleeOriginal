"""Receipt OCR routes: upload an image, review extracted rows, confirm into transactions."""
import logging
from datetime import datetime

from flask import render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user

from models import db, UploadedFile, Account, Transaction
from . import ocr
from .service import extract_and_normalize, ALLOWED_IMAGE_TYPES, MAX_IMAGE_BYTES

logger = logging.getLogger(__name__)


def _user_accounts():
    return (Account.query
            .filter_by(user_id=current_user.id, is_active=True)
            .order_by(Account.name)
            .all())


@ocr.route('/receipt', methods=['GET', 'POST'])
@login_required
def upload_receipt():
    """Step 1: upload a receipt image; on success show the review screen."""
    accounts = _user_accounts()

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not file.filename:
            flash('Please choose a receipt image to upload.', 'error')
            return redirect(url_for('ocr.upload_receipt'))

        ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
        media_type = ALLOWED_IMAGE_TYPES.get(ext)
        if not media_type:
            flash('Unsupported image type. Please upload a PNG, JPG, WEBP, or GIF.', 'error')
            return redirect(url_for('ocr.upload_receipt'))

        image_bytes = file.read()
        if not image_bytes:
            flash('The uploaded file is empty.', 'error')
            return redirect(url_for('ocr.upload_receipt'))
        if len(image_bytes) > MAX_IMAGE_BYTES:
            flash('Image is too large (max 10 MB).', 'error')
            return redirect(url_for('ocr.upload_receipt'))

        rows = extract_and_normalize(image_bytes, media_type)
        if not rows:
            flash('Could not read any line items from that image. '
                  'Try a clearer, well-lit photo or enter the transaction manually.', 'error')
            return redirect(url_for('ocr.upload_receipt'))

        return render_template(
            'ocr/review.html',
            rows=rows,
            accounts=accounts,
            account_id=request.form.get('account_id', ''),
            filename=file.filename,
        )

    return render_template('ocr/upload.html', accounts=accounts)


@ocr.route('/receipt/confirm', methods=['POST'])
@login_required
def confirm_receipt():
    """Step 2: persist the user-reviewed rows as transactions."""
    dates = request.form.getlist('date')
    descriptions = request.form.getlist('description')
    amounts = request.form.getlist('amount')
    account_id = request.form.get('account_id') or None
    filename = request.form.get('filename') or 'receipt'

    # Resolve the (optional) target account, scoped to the current user.
    account = None
    if account_id:
        try:
            account = Account.query.filter_by(
                id=int(account_id), user_id=current_user.id).first()
        except (TypeError, ValueError):
            account = None

    parsed_rows = []
    for raw_date, raw_desc, raw_amount in zip(dates, descriptions, amounts):
        description = (raw_desc or '').strip()
        if not description:
            continue
        try:
            amount = float(str(raw_amount).replace(',', '').replace('$', '').strip())
        except (TypeError, ValueError):
            continue
        try:
            date_value = datetime.strptime((raw_date or '').strip(), '%Y-%m-%d')
        except (TypeError, ValueError):
            date_value = datetime.utcnow()
        parsed_rows.append((date_value, description, amount))

    if not parsed_rows:
        flash('No valid rows to import. Please review the extracted values.', 'error')
        return redirect(url_for('ocr.upload_receipt'))

    try:
        uploaded_file = UploadedFile(
            filename=filename,
            user_id=current_user.id,
            upload_date=datetime.utcnow(),
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
        logger.error(f"Error importing receipt transactions: {str(e)}")
        flash('Could not import the transactions. Please try again.', 'error')
        return redirect(url_for('ocr.upload_receipt'))

    flash(f'Imported {len(parsed_rows)} transaction(s) from your receipt.', 'success')
    return redirect(url_for('main.upload'))
