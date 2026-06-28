"""
Admin routes for subscription management and system administration
Completely isolated from core application features
"""
from flask import render_template, redirect, url_for, flash, request, current_app, abort, send_file, session
from flask_login import current_user, login_required
from sqlalchemy import func
from datetime import datetime, timedelta
import pandas as pd
import os

from . import admin, admin_required
from models import db, User, AdminChartOfAccounts, Account, Transaction, CompanySettings, UploadedFile, BankStatementUpload, Entity
from services.chart_of_accounts import seed_entities, seed_admin_charts
from services.entity_chart_schema import default_entity_id, ensure_entity_chart_schema
from .forms import AdminChartOfAccountsForm, ChartOfAccountsUploadForm, CompanySettingsForm

@admin.route('/charts-of-accounts', methods=['GET'])
@login_required
@admin_required
def charts_of_accounts():
    """Manage system-wide Chart of Accounts"""
    ensure_entity_chart_schema()
    seed_entities()
    seed_admin_charts()
    entities = Entity.query.order_by(Entity.name).all()
    entity_choices = [(e.id, e.name) for e in entities]

    form = AdminChartOfAccountsForm()
    form.entity_id.choices = entity_choices
    upload_form = ChartOfAccountsUploadForm()
    upload_form.entity_id.choices = entity_choices

    selected_entity_id = request.args.get('entity_id', type=int)
    if not selected_entity_id:
        selected_entity_id = default_entity_id()

    query = AdminChartOfAccounts.query
    if selected_entity_id:
        query = query.filter_by(entity_id=selected_entity_id)
    accounts = query.order_by(AdminChartOfAccounts.code).all()

    # Get upload errors from session if they exist
    upload_errors = session.pop('upload_errors', None)

    return render_template('admin/charts_of_accounts.html', 
                         form=form, 
                         upload_form=upload_form,
                         accounts=accounts,
                         entities=entities,
                         selected_entity_id=selected_entity_id,
                         upload_errors=upload_errors)

@admin.route('/charts-of-accounts/upload', methods=['POST'])
@login_required
@admin_required
def upload_chart_of_accounts():
    """Upload Chart of Accounts from Excel file with enhanced error handling"""
    form = ChartOfAccountsUploadForm()
    entities = Entity.query.order_by(Entity.name).all()
    form.entity_id.choices = [(e.id, e.name) for e in entities]
    if form.validate_on_submit():
        try:
            entity_id = form.entity_id.data
            file = form.excel_file.data
            df = pd.read_excel(file)

            # Log the columns found in the Excel file
            current_app.logger.info(f"Excel columns found: {df.columns.tolist()}")

            # Create column mapping dictionary
            column_mapping = {
                'Code': ['Code', 'Account Code', 'AccountCode', 'Account_Code'],
                'Account Name': ['Account Name', 'AccountName', 'Name', 'Account_Name'],
                'Category': ['Category'],
                'Sub Category': ['Sub Category', 'SubCategory', 'Sub_Category'],
                'Description': ['Description', 'Desc'],
                'Links': ['Links', 'Link']
            }

            # Map Excel columns to expected names
            df_columns = {}
            for expected_col, possible_names in column_mapping.items():
                found = False
                for name in possible_names:
                    if name in df.columns:
                        df_columns[expected_col] = name
                        found = True
                        break
                if not found and expected_col in ['Code', 'Account Name', 'Category']:
                    flash(f'Required column {expected_col} not found in Excel file', 'danger')
                    return redirect(url_for('admin.charts_of_accounts', entity_id=form.entity_id.data))

            success_count = 0
            error_count = 0
            skipped_count = 0
            error_details = []

            # Process each row with enhanced validation and error handling
            for idx, row in df.iterrows():
                try:
                    # Get values with proper null handling
                    code = str(row[df_columns['Code']]).strip()
                    link = str(row[df_columns['Links']]).strip() if 'Links' in df_columns else code

                    # Pre-validate required fields
                    if not code:
                        error_details.append({
                            'row': idx + 2,
                            'message': 'Code cannot be empty'
                        })
                        error_count += 1
                        continue

                    name = str(row[df_columns['Account Name']]).strip()
                    if not name:
                        error_details.append({
                            'row': idx + 2,
                            'message': 'Account Name cannot be empty'
                        })
                        error_count += 1
                        continue

                    category = str(row[df_columns['Category']]).strip()
                    if not category:
                        error_details.append({
                            'row': idx + 2,
                            'message': 'Category cannot be empty'
                        })
                        error_count += 1
                        continue

                    # Check for existing account by link or code for this entity
                    existing_account = AdminChartOfAccounts.query.filter(
                        AdminChartOfAccounts.entity_id == entity_id,
                        (AdminChartOfAccounts.link == link) | 
                        (AdminChartOfAccounts.code == code)
                    ).first()

                    if existing_account:
                        error_details.append({
                            'row': idx + 2,
                            'message': f'Account with link {link} or code {code} already exists'
                        })
                        skipped_count += 1
                        continue

                    # Create new account with careful value assignment
                    account = AdminChartOfAccounts(
                        entity_id=entity_id,
                        link=link,
                        code=code,
                        name=name,
                        category=category,
                        sub_category=str(row[df_columns.get('Sub Category', '')]).strip() if 'Sub Category' in df_columns else '',
                        description=str(row[df_columns.get('Description', '')]).strip() if 'Description' in df_columns else ''
                    )

                    db.session.add(account)
                    db.session.commit()  # Commit each record individually
                    success_count += 1

                except Exception as e:
                    db.session.rollback()
                    error_count += 1
                    error_details.append({
                        'row': idx + 2,
                        'message': str(e)
                    })
                    current_app.logger.error(f"Row {idx + 2}: {str(e)}")
                    continue

            if error_details:
                session['upload_errors'] = error_details
                flash('Upload completed with errors. See detailed error report below.', 'warning')
            else:
                session.pop('upload_errors', None)
                flash('Upload completed successfully.', 'success')

            flash(f'Uploaded {success_count} accounts successfully. {error_count} accounts failed. {skipped_count} accounts skipped (already exist).', 'info')

        except Exception as e:
            db.session.rollback()
            flash(f'Error uploading Chart of Accounts: {str(e)}', 'danger')
            current_app.logger.error(f"Error uploading COA: {str(e)}")
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f'{getattr(form, field).label.text}: {error}', 'danger')

    entity_redirect = request.args.get('entity_id', type=int)
    if form.entity_id.data:
        entity_redirect = form.entity_id.data
    return redirect(url_for('admin.charts_of_accounts', entity_id=entity_redirect))

@admin.route('/charts-of-accounts/add', methods=['POST'])
@login_required
@admin_required
def add_chart_of_accounts():
    """Add a new account to system-wide Chart of Accounts"""
    form = AdminChartOfAccountsForm()
    form.entity_id.choices = [(e.id, e.name) for e in Entity.query.order_by(Entity.name).all()]
    if form.validate_on_submit():
        # Check if account code already exists for this entity
        existing_account = AdminChartOfAccounts.query.filter_by(
            entity_id=form.entity_id.data,
            code=form.account_code.data
        ).first()

        if existing_account:
            flash('Account code already exists.', 'error')
            return redirect(url_for('admin.charts_of_accounts', entity_id=form.entity_id.data))

        account = AdminChartOfAccounts(
            entity_id=form.entity_id.data,
            code=form.account_code.data,
            name=form.name.data,
            category=form.category.data,
            sub_category=form.sub_category.data,
            description=form.description.data,
            link=form.account_code.data if not form.link.data else form.link.data
        )
        try:
            db.session.add(account)
            db.session.commit()
            flash('Chart of Accounts entry added successfully.', 'success')
        except Exception as e:
            db.session.rollback()
            flash('Error adding Chart of Accounts entry.', 'error')
            current_app.logger.error(f"Error adding admin COA: {str(e)}")
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f'{getattr(form, field).label.text}: {error}', 'error')

    entity_redirect = form.entity_id.data if form.entity_id.data else request.args.get('entity_id', type=int)
    return redirect(url_for('admin.charts_of_accounts', entity_id=entity_redirect))

@admin.route('/charts-of-accounts/edit/<int:account_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_chart_of_accounts(account_id):
    """Edit an existing Chart of Accounts entry"""
    account = AdminChartOfAccounts.query.get_or_404(account_id)
    form = AdminChartOfAccountsForm()
    form.entity_id.choices = [(e.id, e.name) for e in Entity.query.order_by(Entity.name).all()]

    if request.method == 'GET':
        form.entity_id.data = account.entity_id
        form.account_code.data = account.code
        form.name.data = account.name
        form.category.data = account.category
        form.sub_category.data = account.sub_category
        form.description.data = account.description
        form.link.data = account.link

    if form.validate_on_submit():
        try:
            account.entity_id = form.entity_id.data
            account.code = form.account_code.data
            account.name = form.name.data
            account.category = form.category.data
            account.sub_category = form.sub_category.data
            account.description = form.description.data
            if form.link.data:
                account.link = form.link.data

            db.session.commit()
            flash('Account updated successfully.', 'success')
            return redirect(url_for('admin.charts_of_accounts', entity_id=account.entity_id))
        except Exception as e:
            db.session.rollback()
            flash('Error updating account.', 'error')
            current_app.logger.error(f"Error updating admin COA: {str(e)}")

    return render_template('admin/edit_chart_of_accounts.html', form=form, account=account)

@admin.route('/charts-of-accounts/delete/<int:account_id>', methods=['POST'])
@login_required
@admin_required
def delete_chart_of_accounts(account_id):
    """Delete a Chart of Accounts entry"""
    account = AdminChartOfAccounts.query.get_or_404(account_id)

    try:
        db.session.delete(account)
        db.session.commit()
        flash('Account deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Error deleting account.', 'error')
        current_app.logger.error(f"Error deleting admin COA: {str(e)}")

    return redirect(url_for('admin.charts_of_accounts', entity_id=account.entity_id))


@admin.route('/dashboard')
@login_required
@admin_required
def dashboard():
    """Admin dashboard showing subscription and system statistics"""
    try:
        # Gather subscription statistics
        total_users = User.query.filter(User.is_admin == False).count()
        active_users = User.query.filter(
            User.is_admin == False,
            User.subscription_status == 'active'
        ).count()
        pending_users = User.query.filter(
            User.is_admin == False,
            User.subscription_status == 'pending'
        ).count()
        deactivated_users = User.query.filter(
            User.is_admin == False,
            User.subscription_status == 'deactivated'
        ).count()

        return render_template('admin/dashboard.html',
                           total_users=total_users,
                           active_users=active_users,
                           pending_users=pending_users,
                           deactivated_users=deactivated_users)
    except Exception as e:
        current_app.logger.error(f"Error in admin dashboard: {str(e)}")
        flash('Error loading admin dashboard', 'error')
        return redirect(url_for('auth.login'))

@admin.route('/active-subscribers')
@login_required
@admin_required
def active_subscribers():
    """View and manage active subscribers"""
    try:
        users = User.query.filter(
            User.is_admin == False,
            User.subscription_status == 'active'
        ).all()
        return render_template('admin/active_subscribers.html', users=users)
    except Exception as e:
        current_app.logger.error(f"Error loading active subscribers: {str(e)}")
        flash('Error loading active subscribers', 'error')
        return redirect(url_for('admin.dashboard'))

@admin.route('/deactivated-subscribers')
@login_required
@admin_required
def deactivated_subscribers():
    """View deactivated subscribers"""
    try:
        users = User.query.filter(
            User.is_admin == False,
            User.subscription_status == 'deactivated'
        ).all()
        return render_template('admin/deactivated_subscribers.html', users=users)
    except Exception as e:
        current_app.logger.error(f"Error loading deactivated subscribers: {str(e)}")
        flash('Error loading deactivated subscribers', 'error')
        return redirect(url_for('admin.dashboard'))

@admin.route('/pending-subscribers')
@login_required
@admin_required
def pending_subscribers():
    """View pending subscriber requests"""
    try:
        users = User.query.filter(
            User.is_admin == False,
            User.subscription_status == 'pending'
        ).all()
        return render_template('admin/pending_subscribers.html', users=users)
    except Exception as e:
        current_app.logger.error(f"Error loading pending subscribers: {str(e)}")
        flash('Error loading pending subscribers', 'error')
        return redirect(url_for('admin.dashboard'))

@admin.route('/subscriber/<int:user_id>/approve', methods=['POST'])
@login_required
@admin_required
def approve_subscriber(user_id):
    """Approve a pending subscriber and set up their Chart of Accounts"""
    try:
        user = User.query.get_or_404(user_id)
        if user.is_admin:
            abort(400)  # Bad Request
        # First activate the subscription
        user.subscription_status = 'active'
        # Then create default accounts for the user
        User.create_default_accounts(user.id)
        db.session.commit()
        flash(f'Subscription activated for user {user.username} and Chart of Accounts created', 'success')
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error activating subscription: {str(e)}")
        flash('Error activating subscription', 'error')
    return redirect(url_for('admin.pending_subscribers'))

@admin.route('/subscriber/<int:user_id>/deactivate', methods=['POST'])
@login_required
@admin_required
def deactivate_subscriber(user_id):
    """Deactivate a subscriber"""
    try:
        user = User.query.get_or_404(user_id)
        if user.is_admin:
            abort(400)  # Bad Request
        user.subscription_status = 'deactivated'
        db.session.commit()
        flash(f'Subscription deactivated for user {user.username}', 'success')
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error deactivating subscription: {str(e)}")
        flash('Error deactivating subscription', 'error')
    return redirect(url_for('admin.active_subscribers'))

@admin.route('/subscriber/<int:user_id>/suspend', methods=['POST'])
@login_required
@admin_required
def suspend_subscriber(user_id):
    """Temporarily suspend a subscriber's access"""
    try:
        user = User.query.get_or_404(user_id)
        if user.is_admin:
            abort(400)  # Bad Request
        user.suspend_subscription()
        db.session.commit()
        flash(f'Subscription suspended for user {user.username}', 'success')
        current_app.logger.info(f"Admin {current_user.username} suspended subscription for user {user.username}")
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error suspending subscription: {str(e)}")
        flash('Error suspending subscription', 'error')
    return redirect(url_for('admin.active_subscribers'))

@admin.route('/subscriber/<int:user_id>/reactivate', methods=['POST'])
@login_required
@admin_required
def reactivate_subscriber(user_id):
    """Reactivate a suspended subscriber"""
    try:
        user = User.query.get_or_404(user_id)
        if user.is_admin:
            abort(400)  # Bad Request
        user.reactivate_subscription()
        db.session.commit()
        flash(f'Subscription reactivated for user {user.username}', 'success')
        current_app.logger.info(f"Admin {current_user.username} reactivated subscription for user {user.username}")
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error reactivating subscription: {str(e)}")
        flash('Error reactivating subscription', 'error')
    return redirect(url_for('admin.deactivated_subscribers'))

@admin.route('/subscription-stats')
@login_required
@admin_required
def subscription_stats():
    """View detailed subscription statistics"""
    try:
        # Gather subscription statistics
        total_users = User.query.filter(User.is_admin == False).count()
        active_users = User.query.filter(
            User.is_admin == False,
            User.subscription_status == 'active'
        ).count()
        suspended_users = User.query.filter(
            User.is_admin == False,
            User.subscription_status == 'suspended'
        ).count()
        pending_users = User.query.filter(
            User.is_admin == False,
            User.subscription_status == 'pending'
        ).count()
        deactivated_users = User.query.filter(
            User.is_admin == False,
            User.subscription_status == 'deactivated'
        ).count()

        # Get recent subscription changes
        recent_changes = User.query.filter(
            User.is_admin == False,
            User.updated_at >= (datetime.utcnow() - timedelta(days=7))
        ).order_by(User.updated_at.desc()).limit(10).all()

        return render_template('admin/subscription_stats.html',
                           total_users=total_users,
                           active_users=active_users,
                           suspended_users=suspended_users,
                           pending_users=pending_users,
                           deactivated_users=deactivated_users,
                           recent_changes=recent_changes)
    except Exception as e:
        current_app.logger.error(f"Error loading subscription stats: {str(e)}")
        flash('Error loading subscription statistics', 'error')
        return redirect(url_for('admin.dashboard'))

@admin.route('/subscriber/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_subscriber(user_id):
    """Delete a deactivated subscriber and remove all associated data"""
    try:
        # Verify system is not in maintenance mode
        if current_app.config.get('MAINTENANCE_MODE', False):
            flash('System is in maintenance mode. Deletions are temporarily disabled.', 'warning')
            return redirect(url_for('admin.deactivated_subscribers'))

        user = User.query.get_or_404(user_id)

        # Enhanced safety checks
        if user.is_admin:
            abort(400)  # Bad Request - Cannot delete admin users

        if user.subscription_status != 'deactivated':
            flash('Only deactivated users can be deleted', 'error')
            return redirect(url_for('admin.deactivated_subscribers'))

        # Log the deletion attempt
        current_app.logger.info(f"Admin {current_user.username} deleting user {user.username}")

        try:
            # Mark user as deleted first
            user.soft_delete()

            # Delete user's data in specific order to handle dependencies
            BankStatementUpload.query.filter_by(user_id=user.id).delete()
            db.session.flush()

            Transaction.query.filter_by(user_id=user.id).delete()
            db.session.flush()

            # Delete accounts after transactions are removed
            Account.query.filter_by(user_id=user.id).delete()
            db.session.flush()

            # Delete all other user-related data
            CompanySettings.query.filter_by(user_id=user.id).delete()
            UploadedFile.query.filter_by(user_id=user.id).delete()
            FinancialGoal.query.filter_by(user_id=user.id).delete()
            AlertConfiguration.query.filter_by(user_id=user.id).delete()
            AlertHistory.query.filter_by(user_id=user.id).delete()
            HistoricalData.query.filter_by(user_id=user.id).delete()
            RiskAssessment.query.filter_by(user_id=user.id).delete()
            FinancialRecommendation.query.filter_by(user_id=user.id).delete()
            db.session.flush()

            # Force logout if the deleted user has any active sessions
            from flask_login import logout_user
            if current_user.id == user.id:
                logout_user()

            db.session.commit()
            flash(f'User {user.username} has been permanently deleted', 'success')
            current_app.logger.info(f"User {user.username} deleted successfully by admin {current_user.username}")

        except Exception as db_error:
            db.session.rollback()
            error_msg = str(db_error)
            current_app.logger.error(f"Database error while deleting user {user.username}: {error_msg}")
            flash(f'Database error while deleting user: {error_msg}', 'error')
            return redirect(url_for('admin.deactivated_subscribers'))

    except Exception as e:
        db.session.rollback()
        error_msg = str(e)
        current_app.logger.error(f"Error deleting user: {error_msg}")
        flash(f'Error deleting user: {error_msg}', 'error')

    return redirect(url_for('admin.deactivated_subscribers'))