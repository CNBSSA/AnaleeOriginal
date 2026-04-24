"""
Routes for handling bank statement uploads
Implements secure file handling and validation
"""
import logging
import os
from flask import render_template, request, redirect, url_for, flash, jsonify, current_app
from werkzeug.utils import secure_filename
from flask_login import login_required, current_user

from . import bank_statements
from .forms import BankStatementUploadForm
from .services import BankStatementService
from .reconciliation import ReconciliationService
from models import Account, BankStatementUpload, db, Transaction

# Configure logging
logger = logging.getLogger(__name__)

@bank_statements.route('/reconcile', methods=['POST'])
@login_required
def reconcile():
    """Handle one-click reconciliation request"""
    try:
        # Initialize reconciliation service
        service = ReconciliationService(current_user.id)
        success, result = service.perform_cleanup()

        if success:
            stats = result['cleanup_stats']
            flash(
                f"Reconciliation completed successfully! "
                f"Processed {stats['total_processed']} transactions, "
                f"removed {stats['duplicates_removed']} duplicates, "
                f"fixed {stats['invalid_dates_fixed']} invalid dates.",
                'success'
            )

            # If it's an AJAX request, return JSON response
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'success': True,
                    'stats': stats,
                    'reconciliation_report': result['reconciliation_report']
                })

        else:
            error_msg = result.get('error', 'Reconciliation failed')
            flash(f'Error during reconciliation: {error_msg}', 'error')

            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'success': False,
                    'error': error_msg
                }), 400

        return redirect(url_for('bank_statements.upload'))

    except Exception as e:
        logger.error(f"Error in reconciliation route: {str(e)}", exc_info=True)
        error_msg = 'An unexpected error occurred during reconciliation'
        flash(error_msg, 'error')

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': False,
                'error': error_msg,
                'details': [str(e)]
            }), 500

        return redirect(url_for('bank_statements.upload'))

@bank_statements.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """Handle bank statement upload with enhanced validation and error handling"""
    logger.info(f"Bank statement upload request for user {current_user.id}")
    try:
        form = BankStatementUploadForm()
        logger.info(f"Processing bank statement upload request for user {current_user.id}")

        # Get user's bank accounts
        bank_accounts = Account.query.filter(
            Account.user_id == current_user.id,
            Account.link.like('ca.810%')
        ).order_by(Account.name).all()

        # Update form choices
        form.account.choices = [(acc.id, f"{acc.name} ({acc.link})") for acc in bank_accounts]

        if request.method == 'POST':
            # Check if it's an AJAX request
            is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

            if not form.validate_on_submit():
                error_messages = []
                for field, errors in form.errors.items():
                    error_messages.extend(errors)
                    logger.error(f"Form validation error in {field}: {errors}")

                if is_ajax:
                    return jsonify({
                        'success': False,
                        'error': 'Form validation failed',
                        'details': error_messages
                    }), 400

                for error in error_messages:
                    flash(error, 'error')
                return redirect(url_for('bank_statements.upload'))

            try:
                # Get selected bank account with validation
                account_id = int(form.account.data)
                account = Account.query.filter_by(id=account_id, user_id=current_user.id).first()

                if not account:
                    error_msg = 'Invalid bank account selected'
                    logger.error(f"Invalid account access attempt: {account_id} by user {current_user.id}")
                    if is_ajax:
                        return jsonify({
                            'success': False,
                            'error': error_msg
                        }), 400
                    flash(error_msg, 'error')
                    return redirect(url_for('bank_statements.upload'))

                # Validate file
                file = form.file.data
                if not file or not file.filename:
                    error_msg = 'Please select a file to upload'
                    logger.error("No file selected for upload")
                    if is_ajax:
                        return jsonify({
                            'success': False,
                            'error': error_msg
                        }), 400
                    flash(error_msg, 'error')
                    return redirect(url_for('bank_statements.upload'))

                # Process upload using service
                service = BankStatementService()
                success, response = service.process_upload(
                    file=file,
                    account_id=account_id,
                    user_id=current_user.id
                )

                if success:
                    logger.info(f"Successfully processed upload for user {current_user.id}")
                    if is_ajax:
                        return jsonify(response)
                    flash('Bank statement uploaded and processed successfully!', 'success')
                else:
                    logger.error(f"Upload processing failed: {response.get('error')}")
                    if is_ajax:
                        return jsonify(response), 400
                    flash(response.get('error', 'Upload processing failed'), 'error')
                    if response.get('details'):
                        for detail in response['details']:
                            flash(detail, 'warning')

                return redirect(url_for('bank_statements.upload'))

            except Exception as e:
                error_msg = f"Error processing upload: {str(e)}"
                logger.error(error_msg, exc_info=True)
                if is_ajax:
                    return jsonify({
                        'success': False,
                        'error': 'An error occurred during upload',
                        'details': [str(e)]
                    }), 500
                flash('An error occurred during upload. Please try again.', 'error')
                return redirect(url_for('bank_statements.upload'))

        # Get recent uploads with status
        recent_uploads = BankStatementUpload.query.filter_by(
            user_id=current_user.id
        ).order_by(BankStatementUpload.upload_date.desc()).limit(10).all()

        return render_template('bank_statements/upload.html',
                            form=form,
                            bank_accounts=bank_accounts,
                            recent_uploads=recent_uploads)

    except Exception as e:
        logger.error(f"Error in upload route: {str(e)}", exc_info=True)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': False,
                'error': 'An unexpected error occurred',
                'details': [str(e)]
            }), 500
        flash('An unexpected error occurred. Please try again.', 'error')
        return redirect(url_for('bank_statements.upload'))