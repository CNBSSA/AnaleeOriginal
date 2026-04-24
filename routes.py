"""Main application routes including core functionality"""
import logging
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from flask_login import current_user, login_required, logout_user
from sqlalchemy import text
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from wtforms import FileField, SelectField, SubmitField
from wtforms.validators import DataRequired
from datetime import datetime
from sqlalchemy import func
from models import Transaction
from bank_statements.services import BankStatementService
from predictive_features import PredictiveFeatures


from models import (
    db, User, CompanySettings, Account, Transaction,
    UploadedFile, AdminChartOfAccounts,
    AlertHistory, AlertConfiguration, FinancialGoal,
    FinancialRecommendation, RiskAssessment
)
from forms.company import CompanySettingsForm
from ai_insights import FinancialInsightsGenerator
from maintenance_monitor import MaintenanceMonitor
from alert_system import AlertSystem
from anomaly_detection import AnomalyDetectionService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UploadForm(FlaskForm):
    """Form for handling file uploads with CSRF protection"""
    account = SelectField('Bank Account', validators=[DataRequired()],
                         description='Select the bank account for this statement')
    file = FileField('Statement File', validators=[DataRequired()])
    submit = SubmitField('Upload')

    def __init__(self, *args, **kwargs):
        """Initialize form with user's bank accounts"""
        super(UploadForm, self).__init__(*args, **kwargs)
        if current_user.is_authenticated:
            try:
                accounts = Account.query.filter(
                    Account.user_id == current_user.id,
                    #Account.link.like('ca.810%'),
                    #Account.is_active == True
                ).order_by(Account.link).all()

                logger.info(f"Found {len(accounts)} active bank accounts for user {current_user.id}")
                self.account.choices = [(str(acc.id), f"{acc.link} - {acc.name}") 
                                     for acc in accounts]

                if not self.account.choices:
                    logger.warning(f"No active bank accounts found for user {current_user.id}")
            except Exception as e:
                logger.error(f"Error loading bank accounts: {str(e)}")
                self.account.choices = []

main = Blueprint('main', __name__)

@main.route('/')
def index():
    """Root route - redirects to appropriate dashboard based on authentication status"""
    try:
        if current_user.is_authenticated:
            if current_user.is_deleted:
                logout_user()
                flash('This account has been deleted. Please contact support to restore your account.', 'error')
                return redirect(url_for('auth.login'))

            if current_user.is_admin:
                return redirect(url_for('admin.dashboard'))
            return redirect(url_for('main.dashboard'))

        return redirect(url_for('auth.login'))
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return redirect(url_for('auth.login'))

@main.route('/dashboard')
@login_required
def dashboard():
    # Get current month's transactions
    current_month = datetime.now().month
    current_year = datetime.now().year
    
    # Calculate total income and expenses
    monthly_transactions = Transaction.query.filter(
        Transaction.user_id == current_user.id,
        func.extract('month', Transaction.date) == current_month,
        func.extract('year', Transaction.date) == current_year
    ).all()
    
    total_income = sum(t.amount for t in monthly_transactions if t.amount > 0)
    total_expenses = sum(abs(t.amount) for t in monthly_transactions if t.amount < 0)
    
    # Get last 6 months labels and data
    monthly_labels = []
    monthly_income = []
    monthly_expenses = []
    
    for i in range(5, -1, -1):
        month = (current_month - i) if current_month > i else (12 - (i - current_month))
        year = current_year if current_month > i else current_year - 1
        month_name = datetime(year, month, 1).strftime('%B')
        monthly_labels.append(month_name)
        
        month_transactions = Transaction.query.filter(
            Transaction.user_id == current_user.id,
            func.extract('month', Transaction.date) == month,
            func.extract('year', Transaction.date) == year
        ).all()
        
        income = sum(t.amount for t in month_transactions if t.amount > 0)
        expenses = sum(abs(t.amount) for t in month_transactions if t.amount < 0)
        monthly_income.append(income)
        monthly_expenses.append(expenses)
    
    return render_template('dashboard.html',
        total_income=total_income,
        total_expenses=total_expenses,
        monthly_labels=monthly_labels,
        monthly_income=monthly_income,
        monthly_expenses=monthly_expenses,
        transactions=monthly_transactions
    )

@main.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    """Protected Chart of Accounts management"""
    try:
        if request.method == 'POST':
            account = Account(
                link=request.form['link'],
                name=request.form['name'],
                category=request.form['category'],
                sub_category=request.form.get('sub_category', ''),
                account_code=request.form.get('account_code', ''),
                user_id=current_user.id
            )
            db.session.add(account)
            db.session.commit()
            flash('Account added successfully', 'success')
            logger.info(f'New account added: {account.name}')

        # Get user's accounts
        accounts = Account.query.filter_by(
            user_id=current_user.id,
            is_active=True
        ).all()

        # Get system-wide Chart of Accounts for reference
        system_accounts = AdminChartOfAccounts.query.all()

        return render_template(
            'settings.html',
            accounts=accounts,
            system_accounts=system_accounts
        )
    except Exception as e:
        logger.error(f'Error in settings route: {str(e)}')
        db.session.rollback()
        flash('Error accessing Chart of Accounts', 'error')
        return redirect(url_for('main.dashboard'))

@main.route('/settings/import-charts', methods=['GET'])
@login_required
def import_chart_of_accounts():
    admin_accounts = AdminChartOfAccounts.query.all()
    for acc in admin_accounts:
        account = Account(
            link=acc.link,
            name=acc.name,
            category=acc.category,
            sub_category=acc.sub_category,
            account_code=acc.code,
            user_id=current_user.id
        )
        db.session.add(account)
    db.session.commit()
    flash('Chart of Accounts imported successfully', 'success')
    return redirect(url_for('main.settings'))


@main.route('/company-settings', methods=['GET', 'POST'])
@login_required
def company_settings():
    """Handle company settings with CSRF protection"""
    form = CompanySettingsForm()
    settings = CompanySettings.query.filter_by(user_id=current_user.id).first()

    if request.method == 'POST' and form.validate_on_submit():
        try:
            if not settings:
                settings = CompanySettings(user_id=current_user.id)
                db.session.add(settings)

            settings.company_name = form.company_name.data
            settings.registration_number = form.registration_number.data
            settings.tax_number = form.tax_number.data
            settings.vat_number = form.vat_number.data
            settings.address = form.address.data
            settings.financial_year_end = int(form.financial_year_end.data)

            db.session.commit()
            flash('Company settings updated successfully', 'success')

        except Exception as e:
            logger.error(f'Error updating company settings: {str(e)}')
            flash('Error updating company settings', 'error')
            db.session.rollback()

    if settings:
        form.company_name.data = settings.company_name
        form.registration_number.data = settings.registration_number
        form.tax_number.data = settings.tax_number
        form.vat_number.data = settings.vat_number
        form.address.data = settings.address
        form.financial_year_end.data = str(settings.financial_year_end)

    months = [
        (1, 'January'), (2, 'February'), (3, 'March'),
        (4, 'April'), (5, 'May'), (6, 'June'),
        (7, 'July'), (8, 'August'), (9, 'September'),
        (10, 'October'), (11, 'November'), (12, 'December')
    ]

    return render_template(
        'company_settings.html',
        form=form,
        settings=settings,
        months=months
    )

@main.route('/analyze')
@login_required
def analyze_list():
    """Show list of files available for analysis"""
    try:
        files = UploadedFile.query.filter_by(user_id=current_user.id).order_by(UploadedFile.upload_date.desc()).all()
        return render_template('analyze_list.html', files=files)
    except Exception as e:
        logger.error(f"Error loading files for analysis: {str(e)}")
        flash('Error loading files', 'error')
        return redirect(url_for('main.upload'))

@main.route('/analyze/<int:file_id>')
@login_required
def analyze(file_id):
    """Analyze specific file transactions with enhanced processing checks"""
    logger.info(f"Starting analysis for file_id: {file_id} for user {current_user.id}")

    try:
        # Verify database connection first
        try:
            db.session.execute(text('SELECT 1'))
            logger.info("Database connection verified")
        except Exception as db_error:
            logger.error(f"Database connection error: {str(db_error)}")
            db.session.rollback()
            flash('Unable to connect to database. Please try again.')
            return redirect(url_for('main.upload'))

        # Load file and verify ownership with detailed logging
        file = UploadedFile.query.filter_by(id=file_id, user_id=current_user.id).first()
        logger.info(f"Database query completed. File found: {file is not None}")

        if not file:
            logger.error(f"File {file_id} not found or unauthorized access for user {current_user.id}")
            flash('File not found or unauthorized access')
            return redirect(url_for('main.upload'))

        # Load transactions with improved filtering
        try:
            transactions = Transaction.query.filter_by(
                file_id=file_id,
                user_id=current_user.id
            ).order_by(Transaction.date).all()

            logger.info(f"Found {len(transactions)} total transactions for file {file_id}")

            if not transactions:
                logger.warning(f"No transactions found for file {file_id}")
                flash('No transactions found in this file')
                return redirect(url_for('main.upload'))

            # Load accounts for the user
            accounts = Account.query.filter_by(
                user_id=current_user.id,
                is_active=True
            ).all()

            if not accounts:
                logger.warning(f"No active accounts found for user {current_user.id}")
                #flash('Please set up your Chart of Accounts first')
                #return redirect(url_for('main.settings'))

            logger.info(f"Successfully loaded {len(accounts)} active accounts")

            # Initialize insights with improved processing logic
            transaction_insights = {}
            for transaction in transactions:
                # Only mark as needing processing if no account AND no explanation
                needs_processing = (
                    transaction.account_id is None and 
                    (transaction.explanation is None or transaction.explanation.strip() == '')
                )

                transaction_insights[transaction.id] = {
                    'similar_transactions': [],
                    'pattern_matches': [],
                    'keyword_matches': [],
                    'rule_matches': [],
                    'explanation_suggestion': None,
                    'confidence': 0,
                    'ai_processed': False,
                    'needs_processing': needs_processing
                }

            # Count unprocessed transactions with improved criteria
            unprocessed_count = sum(1 for t in transactions 
                                  if transaction_insights[t.id]['needs_processing'])

            logger.info(f"Found {unprocessed_count} transactions that need processing")

            if unprocessed_count == 0:
                flash('All transactions have been processed')
            else:
                flash(f'Found {unprocessed_count} transactions that need processing')

            return render_template(
                'analyze.html',
                file=file,
                transactions=transactions,
                accounts=accounts,
                transaction_insights=transaction_insights,
                unprocessed_count=unprocessed_count,
                ai_available=True
            )

        except Exception as tx_error:
            logger.error(f"Error loading transactions: {str(tx_error)}")
            logger.exception("Full transaction loading error:")
            db.session.rollback()
            flash('Error loading transaction data. Please try again.')
            return redirect(url_for('main.upload'))

    except Exception as e:
        logger.error(f"Error in analyze route: {str(e)}")
        logger.exception("Full analyze route error:")
        flash('Error loading transaction data')
        return redirect(url_for('main.upload'))
    
    
@main.route('/analyze/save-transaction/<int:transaction_id>', methods=['POST'])
@login_required
def save_transaction(transaction_id):
    """Save transaction details with enhanced error handling"""
    try:
        data = request.get_json()
        account_id = data.get('account_id', type=int)
        explanation = data.get('explanation', '').strip()

        if not account_id:
            return jsonify({'error': 'Account is required'}), 400

        transaction = Transaction.query.filter_by(
            id=transaction_id,
            user_id=current_user.id
        ).first()

        if not transaction:
            return jsonify({'error': 'Transaction not found'}), 404

        transaction.account_id = account_id
        transaction.explanation = explanation
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Transaction details saved successfully'
        })

    except Exception as e:
        logger.error(f"Error saving transaction: {str(e)}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@main.route('/analyze/similar-transactions', methods=['POST'])
@login_required
def find_similar_transactions_api():
    """ERF: Find similar transactions based on description and explanation"""
    try:
        data = request.get_json()
        description = data.get('description', '').strip()
        explanation = data.get('explanation', '').strip()

        if not description:
            return jsonify({'error': 'Description is required'}), 400

        predictor = PredictiveFeatures()
        similar_transactions = predictor.find_similar_transactions(description, explanation)

        return jsonify({
            'success': True,
            'similar_transactions': similar_transactions
        })

    except Exception as e:
        logger.error(f"Error finding similar transactions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@main.route('/analyze/suggest-account', methods=['POST'])
@login_required
def suggest_account():
    """ASF: Suggest account based on description and explanation"""
    try:
        data = request.get_json()
        description = data.get('description', '').strip()
        explanation = data.get('explanation', '').strip()

        if not description:
            return jsonify({'error': 'Description is required'}), 400

        predictor = PredictiveFeatures()
        suggestion = predictor.suggest_account(description, explanation)

        return jsonify(suggestion)

    except Exception as e:
        logger.error(f"Error suggesting account: {str(e)}")
        return jsonify({'error': str(e)}), 500

@main.route('/analyze/suggest-explanation', methods=['POST'])
@login_required
def suggest_explanation():
    """ESF: Suggest explanation based on transaction description"""
    try:
        data = request.get_json()
        description = data.get('description', '').strip()

        if not description:
            return jsonify({'error': 'Description is required'}), 400

        predictor = PredictiveFeatures()
        suggestion = predictor.suggest_explanation(description)

        return jsonify(suggestion)

    except Exception as e:
        logger.error(f"Error suggesting explanation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@main.route('/account/<int:account_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_account(account_id):
    try:
        account = Account.query.get_or_404(account_id)
        if account.user_id != current_user.id:
            flash('Access denied', 'error')
            return redirect(url_for('main.settings'))

        if request.method == 'POST':
            account.link = request.form['link']
            account.name = request.form['name']
            account.category = request.form['category']
            account.sub_category = request.form.get('sub_category', '')
            try:
                db.session.commit()
                flash('Account updated successfully', 'success')
                return redirect(url_for('main.settings'))
            except Exception as e:
                logger.error(f'Error updating account: {str(e)}')
                flash(f'Error updating account: {str(e)}', 'error')
                db.session.rollback()

        return render_template('edit_account.html', account=account)
    except Exception as e:
        logger.error(f'Error in edit_account route: {str(e)}')
        flash('Error accessing account', 'error')
        return redirect(url_for('main.settings'))

@main.route('/account/<int:account_id>/delete', methods=['POST'])
@login_required
def delete_account(account_id):
    account = Account.query.get_or_404(account_id)
    if account.user_id != current_user.id:
        flash('Access denied')
        return redirect(url_for('main.settings'))

    try:
        db.session.delete(account)
        db.session.commit()
        flash('Account deleted successfully')
    except Exception as e:
        logger.error(f'Error deleting account: {str(e)}')
        flash(f'Error deleting account: {str(e)}')
        db.session.rollback()
    return redirect(url_for('main.settings'))


@main.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """Handle file uploads with comprehensive error handling and validation"""
    try:
        form = UploadForm()

        # Get uploaded files with detailed logging
        files = UploadedFile.query.filter_by(user_id=current_user.id).order_by(UploadedFile.upload_date.desc()).all()
        logger.info(f"Retrieved {len(files)} existing files for user {current_user.id}")

        if request.method == 'POST':
            logger.info("Processing upload request")

            # Form validation with CSRF protection
            if not form.validate_on_submit():
                logger.error(f"Form validation failed: {form.errors}")
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'success': False, 'error': 'Form validation failed'}), 400
                flash('Please ensure all fields are filled correctly', 'error')
                return redirect(url_for('main.upload'))

            try:
                file = form.file.data
                if not file or not file.filename:
                    logger.warning("No file selected")
                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        return jsonify({'success': False, 'error': 'No file selected'}), 400
                    flash('No file selected', 'error')
                    return redirect(url_for('main.upload'))

                # Secure the filename
                filename = secure_filename(file.filename)

                # Validate file format
                if not filename.lower().endswith(('.csv', '.xlsx')):
                    logger.warning(f"Invalid file format: {filename}")
                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        return jsonify({'success': False, 'error': 'Invalid file format'}), 400
                    flash('Invalid file format. Please upload a CSV or Excel file.', 'error')
                    return redirect(url_for('main.upload'))
                
                bank_statement_service = BankStatementService()
                account_id = int(form.account.data)
                success, response = bank_statement_service.process_upload(
                    file=file,
                    account_id=account_id,
                    user_id=current_user.id
                )

                # Create upload record
                # uploaded_file = UploadedFile(
                #     filename=filename,
                #     user_id=current_user.id,
                #     upload_date=datetime.utcnow()
                # )
                # db.session.add(uploaded_file)
                # db.session.commit()
                logger.info(f"File upload record created: {filename}")
                
                

                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'success': True,
                        'message': 'File uploaded successfully',
                        #'file_id': uploaded_file.id
                    })

                flash('File uploaded successfully')
                return redirect(url_for('main.upload'))

            except Exception as e:
                logger.error(f"Error processing upload: {str(e)}")
                db.session.rollback()
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'success': False, 'error': str(e)}), 500
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(url_for('main.upload'))

        # GET request - render upload form
        return render_template('upload.html', form=form, files=files)

    except Exception as e:
        logger.error(f"Unexpected error in upload route: {str(e)}")
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'error': 'An unexpected error occurred'}), 500
        flash('An unexpected error occurred')
        return redirect(url_for('main.upload'))

@main.route('/file/<int:file_id>/delete', methods=['POST'])
@login_required
def delete_file(file_id):
    try:
        file = UploadedFile.query.filter_by(id=file_id, user_id=current_user.id).first_or_404()
        Transaction.query.filter_by(file_id=file.id).delete()
        db.session.delete(file)
        db.session.commit()
        flash('File and associated transactions deleted successfully')
        return redirect(url_for('main.upload'))
    except Exception as e:
        logger.error(f'Error deleting file: {str(e)}')
        db.session.rollback()
        flash('Error deleting file')
        return redirect(url_for('main.upload'))

@main.route('/update_explanation', methods=['POST'])
@login_required
def update_explanation():
    try:
        data = request.get_json()
        transaction_id = data.get('transaction_id')
        explanation = data.get('explanation', '').strip()
        description = data.get('description', '').strip()

        if not transaction_id or not description:
            return jsonify({'error': 'Missing required fields'}), 400

        transaction = Transaction.query.filter_by(
            id=transaction_id,
            user_id=current_user.id
        ).first()

        if not transaction:
            return jsonify({'error': 'Transaction not found'}), 404

        transaction.explanation = explanation
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Explanation updated successfully'
        })

    except Exception as e:
        logger.error(f"Error updating explanation: {str(e)}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@main.route('/predict_account', methods=['POST'])
@login_required
def predict_account_route():
    try:
        data = request.get_json()
        description = data.get('description', '').strip()
        explanation = data.get('explanation', '').strip()

        if not description:
            return jsonify({'error': 'Description is required'}), 400

        available_accounts = Account.query.filter_by(
            user_id=current_user.id,
            is_active=True
        ).all()

        if not available_accounts:
            return jsonify({'error': 'No active accounts found'}), 400

        account_data = [{
            'name': acc.name,
            'category': acc.category,
            'balance': 0  # Initialize with zero balance
        } for acc in available_accounts]

        return jsonify({
            'success': True,
            'accounts': account_data
        })

    except Exception as e:
        logger.error(f"Error predicting account: {str(e)}")
        return jsonify({'error': str(e)}), 500

@main.route('/financial-insights')
@login_required
def financial_insights():
    """Financial insights view showing detailed analysis"""
    try:
        company_settings = CompanySettings.query.filter_by(user_id=current_user.id).first()
        if not company_settings:
            flash('Please configure company settings first.')
            return redirect(url_for('main.company_settings'))

        # Get current financial year dates
        fy_dates = company_settings.get_financial_year()
        start_date = fy_dates['start_date']
        end_date = fy_dates['end_date']

        # Get transactions for analysis
        transactions = Transaction.query.filter(
            Transaction.user_id == current_user.id,
            Transaction.date.between(start_date, end_date)
        ).order_by(Transaction.date.desc()).all()

        # Initialize financial insights generator
        insights_generator = FinancialInsightsGenerator()

        # Get AI-generated insights if available
        financial_advice = session.get('financial_advice', {
            'key_insights': [],
            'risk_factors': [],
            'optimization_opportunities': [],
            'strategic_recommendations': [],
            'cash_flow_analysis': {
                'current_status': '',
                'projected_trend': '',
                'key_drivers': [],
                'improvement_suggestions': []
                    }
        })

        return render_template('financial_insights.html',
                             transactions=transactions,
                             start_date=start_date,
                             end_date=end_date,
                             financial_advice=financial_advice)

    except Exception as e:
        logger.error(f"Error in financialinsights: {str(e)}")
        flash('Error generating financial insights')
        return redirect(url_for('main.dashboard'))

@main.route('/generate-insights', methods=['POST'])
@login_required
def generate_insights():
    """Generate AI-powered financial insights"""
    try:
        # Get transactions for the current financial year
        company_settings = CompanySettings.query.filter_by(user_id=current_user.id).first()
        if not company_settings:
            flash('Please configure company settings first.')
            return redirect(url_for('main.company_settings'))

        # Get current financial year dates
        fy_dates = company_settings.get_financial_year()

        # Get transactions for analysis
        transactions = Transaction.query.filter(
            Transaction.user_id == current_user.id,
            Transaction.date.between(fy_dates['start_date'], fy_dates['end_date'])
        ).order_by(Transaction.date.desc()).all()

        # Convert transactions to dictionary format for AI processing
        transaction_data = [{
            'date': t.date.isoformat(),
            'description': t.description,
            'amount': float(t.amount),
            'category': t.account.category if t.account else 'Uncategorized'
        } for t in transactions]

        # Generate insights using AI
        insights_generator = FinancialInsightsGenerator()
        insights = insights_generator.generate_transaction_insights(transaction_data)

        if insights.get('success'):
            # Parse AI response and structure it
            financial_advice = {
                'key_insights': _parse_insights(insights['insights']),
                'risk_factors': _extract_risk_factors(insights['insights']),
                'optimization_opportunities': _extract_opportunities(insights['insights']),
                'strategic_recommendations': _extract_recommendations(insights['insights']),
                'cash_flow_analysis': _analyze_cash_flow(transaction_data)
            }
            session['financial_advice'] = financial_advice
            flash('Financial insights generated successfully', 'success')
        else:
            flash('Unable to generate insights at this time', 'error')

        return redirect(url_for('main.financial_insights'))

    except Exception as e:
        logger.error(f"Error generating AI insights: {str(e)}")
        flash('Error generating financial insights')
        return redirect(url_for('main.financial_insights'))

def _parse_insights(insights_text):
    """Return the AI insights text for display."""
    try:
        if not insights_text:
            return "No insights available."
        return insights_text
    except Exception as e:
        logger.error(f"Error parsing insights: {str(e)}")
        return "Unable to parse insights."

def _extract_risk_factors(insights_text):
    """Extract risk factor bullet points from AI insights text."""
    try:
        if not insights_text:
            return []
        lines = insights_text.split('\n')
        risk_factors = []
        in_risk_section = False
        for line in lines:
            line = line.strip()
            lower = line.lower()
            if any(kw in lower for kw in ('risk', 'concern', 'warning', 'caution', 'threat')):
                in_risk_section = True
            if in_risk_section and line.startswith(('-', '•', '*', '·')) and len(line) > 3:
                risk_factors.append(line.lstrip('-•*· '))
            elif in_risk_section and line and not line.startswith(('-', '•', '*', '·')):
                in_risk_section = False
        return risk_factors if risk_factors else [insights_text[:200]] if insights_text else []
    except Exception:
        return []

def _extract_opportunities(insights_text):
    """Extract optimization opportunity bullet points from AI insights text."""
    try:
        if not insights_text:
            return []
        lines = insights_text.split('\n')
        opportunities = []
        in_opp_section = False
        for line in lines:
            line = line.strip()
            lower = line.lower()
            if any(kw in lower for kw in ('opportunit', 'optimiz', 'improve', 'saving', 'growth', 'potential')):
                in_opp_section = True
            if in_opp_section and line.startswith(('-', '•', '*', '·')) and len(line) > 3:
                opportunities.append(line.lstrip('-•*· '))
            elif in_opp_section and line and not line.startswith(('-', '•', '*', '·')):
                in_opp_section = False
        return opportunities if opportunities else []
    except Exception:
        return []

def _extract_recommendations(insights_text):
    """Extract strategic recommendation bullet points from AI insights text."""
    try:
        if not insights_text:
            return []
        lines = insights_text.split('\n')
        recommendations = []
        in_rec_section = False
        for line in lines:
            line = line.strip()
            lower = line.lower()
            if any(kw in lower for kw in ('recommend', 'suggest', 'action', 'strategy', 'consider', 'should')):
                in_rec_section = True
            if in_rec_section and line.startswith(('-', '•', '*', '·')) and len(line) > 3:
                recommendations.append(line.lstrip('-•*· '))
            elif in_rec_section and line and not line.startswith(('-', '•', '*', '·')):
                in_rec_section = False
        return recommendations if recommendations else []
    except Exception:
        return []

def _analyze_cash_flow(transaction_data):
    """Analyze cash flow patterns from transaction data"""
    try:
        total_inflow = sum(t['amount'] for t in transaction_data if t['amount'] > 0)
        total_outflow = abs(sum(t['amount'] for t in transaction_data if t['amount']< 0))
        net_flow = total_inflow - total_outflow

        return {
            'current_status': f"Net cash flow: ${net_flow:,.2f}",
            'projected_trend': "Trend analysis will be available in the next update",
            'key_drivers': [
                f"Total inflow: ${total_inflow:,.2f}",
                f"Total outflow: ${total_outflow:,.2f}"
            ],
            'improvement_suggestions': ["Cash flow optimization suggestions will be available in the next update"]
        }
    except Exception as e:
        logger.error(f"Error analyzing cash flow: {str(e)}")
        return {
            'current_status': "Unable to analyze cash flow",
            'projected_trend': "Analysis unavailable",
            'key_drivers': [],
            'improvement_suggestions': []
        }

@main.route('/expense-forecast')
@login_required
def expense_forecast():
    """Handle expense forecastview with proper error handling and data structure"""
    try:
        # Initialize the forecast structure with required attributes
        forecast = {
            'confidence_metrics': {
                'overall_confidence': 0.85,
                'variance_range': {
                    'min': 0.0,
                    'max': 0.0
                },
                'reliability_score': 0.80
            },
            'forecast_factors': {
                'key_drivers': []
            },
            'recommendations': []
        }

        # Get all transactions for analysis
        transactions = Transaction.query.filter_by(user_id=current_user.id).order_by(Transaction.date.desc()).all()

        if not transactions:
            flash('No transaction data available for forecasting')
            return redirect(url_for('main.dashboard'))

        # Process transaction data for monthly analysis
        monthly_data = {}
        for transaction in transactions:
            month_key = transaction.date.strftime('%Y-%m')
            if month_key not in monthly_data:
                monthly_data[month_key] = {'amount': 0, 'count': 0}
            monthly_data[month_key]['amount'] += transaction.amount
            monthly_data[month_key]['count'] += 1

        # Prepare data for charts
        sorted_months = sorted(monthly_data.keys())
        monthly_labels = [datetime.strptime(m, '%Y-%m').strftime('%b %Y') for m in sorted_months]
        monthly_amounts = [monthly_data[m]['amount'] for m in sorted_months]
        # Calculate confidence intervals
        import statistics
        mean_amount = sum(monthly_amounts) / len(monthly_amounts) if monthly_amounts else 0
        std_dev = statistics.stdev(monthly_amounts) if len(monthly_amounts) > 1 else 0
        confidence_upper = [amount + std_dev for amount in monthly_amounts]
        confidence_lower = [amount - std_dev for amount in monthly_amounts]

        # Update variance range in forecast
        if monthly_amounts:
            forecast['confidence_metrics']['variance_range'] = {
                'min': min(monthly_amounts),
                'max': max(monthly_amounts)
            }

        # Process category data
        category_data = {}
        for transaction in transactions:
            if transaction.account:
                category = transaction.account.category or 'Uncategorized'
                if category not in category_data:
                    category_data[category] = 0
                category_data[category] += transaction.amount

        category_labels = list(category_data.keys())
        category_amounts = [category_data[cat] for cat in category_labels]

        return render_template('expense_forecast.html',
                             forecast=forecast,
                             monthly_labels=monthly_labels,
                             monthly_amounts=monthly_amounts,
                             confidence_upper=confidence_upper,
                             confidence_lower=confidence_lower,
                             category_labels=category_labels,
                             category_amounts=category_amounts)

    except Exception as e:
        logger.error(f"Error in expense forecast: {str(e)}")
        flash('Error generating expense forecast')
        return redirect(url_for('main.dashboard'))

@main.route('/upload-progress')
@login_required
def upload_progress():
    """Get the current upload progress"""
    progress = session.get('upload_progress', {})
    return jsonify(progress)

@main.route('/icountant', methods=['GET', 'POST'])
@login_required
def icountant_interface():
    """Handle the iCountant interface with proper transaction processing and AI suggestions"""
    logger.info(f"Starting iCountant interface for user {current_user.id}")

    try:
        # Get total transaction count
        total_count = Transaction.query.filter_by(user_id=current_user.id).count()
        logger.info(f"Total transactions for user{current_user.id}: {total_count}")

        # Get unprocessed transactions
        transactions = Transaction.query.filter_by(
            user_id=current_user.id,
            account_id=None
        ).order_by(Transaction.date).all()

        # Get active accounts for processing
        accounts = Account.query.filter_by(
            user_id=current_user.id,
            is_active=True
        ).order_by(Account.category, Account.name).all()
        logger.info(f"Found {len(accounts)} active accounts for user {current_user.id}")

        if request.method == 'POST':
            transaction_id = request.form.get('transaction_id', type=int)
            selected_account = request.form.get('selected_account', type=int)

            if transaction_id and selected_account is not None:
                transaction = Transaction.query.get(transaction_id)
                if transaction and transaction.user_id == current_user.id:
                    if 0 <= selected_account < len(accounts):
                        # Store the AI suggestion before updating
                        if transaction.ai_category:
                            transaction.account_id = accounts[selected_account].id
                            transaction.ai_confidence = float(transaction.ai_confidence or 0.0)
                            db.session.commit()
                            flash('Transaction processed successfully')
                            return redirect(url_for('main.icountant_interface'))

        # Get current transaction and process it
        current_transaction = next((t for t in transactions), None)
        if current_transaction:
            logger.info(f"Processing transaction {current_transaction.id}: {current_transaction.description}")

            # Initialize insights generator
            insights_generator = FinancialInsightsGenerator()

            # Generate insights with AI categorization
            insights = insights_generator.generate_transaction_insights([{
                'date': current_transaction.date.isoformat(),
                'description': current_transaction.description,
                'amount': float(current_transaction.amount),
                'category': current_transaction.account.category if current_transaction.account else None
            }])

            # Store AI suggestions in the transaction
            current_transaction.ai_category = insights['category_suggestion']['category']
            current_transaction.ai_confidence = insights['category_suggestion']['confidence']
            current_transaction.ai_explanation = insights['category_suggestion']['explanation']
            db.session.commit()

            # Format suggestions for display
            suggested_accounts = []
            if insights['category_suggestion']['category']:
                matching_accounts = [acc for acc in accounts 
                                  if acc.category.lower() == insights['category_suggestion']['category'].lower()]
                suggested_accounts = [{
                    'account': acc,
                    'confidence': insights['category_suggestion']['confidence'],
                    'reason': insights['category_suggestion']['explanation']
                } for acc in matching_accounts[:3]]  # Top 3 matching accounts

            transaction_info = {
                'insights': {
                    'amount_formatted': f"${abs(current_transaction.amount):,.2f}",
                    'transaction_type': 'credit' if current_transaction.amount < 0 else 'debit',
                    'ai_insights': insights['insights'],
                    'suggested_accounts': suggested_accounts or [{
                        'account': account,
                        'reason': 'Alternative suggestion based on category'
                    } for account in accounts[:3]]  # Fallback to first 3 accounts if no AI matches
                }
            }
            message = None
        else:
            current_transaction = None
            transaction_info = None
            message = "No transactions pending for processing"

        # Get recently processed transactions        
        recently_processed = Transaction.query.filter(            Transaction.user_id == current_user.id,            Transaction.account_id.isnot(None)
        ).order_by(Transaction.date.desc()).limit(5).all()
        
        processed_count = Transaction.query.filter(
            Transaction.user_id == current_user.id,
            Transaction.account_id.isnot(None)
        ).count()
        
        return render_template(
            'icountant.html',
            transaction=current_transaction,
            transaction_info=transaction_info,
            accounts=accounts,
            message=message,
            recently_processed=recently_processed,
            processed_count=processed_count,
            total_count=total_count
        )
        

    except Exception as e:
        logger.error(f"Error in iCountant interface: {str(e)}")
        db.session.rollback()
        flash('Error processing transaction')
        return redirect(url_for('main.dashboard'))
        
class ICountant:
    def __init__(self, accounts):
        self.accounts = accounts
        
    def process_transaction(self, transaction_data):
        # Placeholder for actual iCountant logic
        message = "iCountant is processing this transaction. Please select an account."
        transaction_info = {}
        return message, transaction_info
        
    def complete_transaction(self, selected_account_index):
        # Placeholder for actual iCountant logic
        if 0 <= selected_account_index < len(self.accounts):
            message = "Transaction completed successfully!"
            success = True
            completed = True
        else:
            message = "Invalid account selection."
            success = False
            completed = False
        return success, message, completed
        
def suggest_explanation(description, similar_transactions):
    #Implementation for suggestion would go here. Placeholder for now.
    return "Explanation suggestion will be available in the next update"
    
def process_uploaded_file(file, status):
    """Process the uploaded file and return dataframe and total rows."""
    import pandas as pd
    try:
        if file.filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            raise ValueError('Invalid file format')
            
        total_rows = len(df)
        return df, total_rows
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise
        
def init_upload_status(filename):
    """Initialize the upload status dictionary."""
    return {
        'status': 'processing',
        'filename': filename,
        'total_rows': 0,
        'processed_rows': 0,
        'current_chunk': 0,
        'progress': 0,
        'last_update': datetime.utcnow().isoformat(),
        'errors': []
    }
    
def process_transaction_rows(df, uploaded_file, user):
    """Process transaction rows from dataframe."""
    processed_rows = 0
    error_rows = []
    
    try:
        for index, row in df.iterrows():
            try:
                transaction = Transaction(
                    date=pd.to_datetime(row['Date']).date(),
                    description=str(row['Description']),
                    amount=float(row['Amount']),
                    file_id=uploaded_file.id,
                    user_id=user.id
                )
                db.session.add(transaction)
                processed_rows += 1
            except Exception as e:
                error_rows.append({
                    'row': index + 2,  # +2 for Excel row number (header + 1-based index)
                    'error': str(e)
                })
                
        db.session.commit()
        return processed_rows, error_rows
        
    except Exception as e:
        logger.error(f"Error processing transactions: {str(e)}")
        db.session.rollback()
        raise
        
from predictive_utils import find_similar_transactions, TEXT_THRESHOLD, SEMANTIC_THRESHOLD
        
@main.route('/api/generate-insights', methods=['POST'])
@login_required
def generate_insights_api():
    try:
        # Get transactions for the current financial year
        company_settings = CompanySettings.query.filter_by(user_id=current_user.id).first()
        if not company_settings:
            return jsonify({'error': 'Please configure company settings first.'}), 400
            
        fy_dates = company_settings.get_financial_year()
        transactions = Transaction.query.filter(
            Transaction.user_id == current_user.id,
            Transaction.date.between(fy_dates['start_date'], fy_dates['end_date'])
        ).order_by(Transaction.date.desc()).all()
        
        # Format transactions for response
        transaction_data = [{
            'id': t.id,
            'date': t.date.strftime('%Y-%m-%d'),
            'description': t.description,
            'amount': float(t.amount),
            'category': t.account.category if t.account else 'Uncategorized'
        } for t in transactions]
        
        # Generate insights using AI
        insights_generator = FinancialInsightsGenerator()
        insights = insights_generator.generate_insights(transaction_data)
        
        return jsonify({
            'transactions': transaction_data,
            'insights': insights
        })
        
    except Exception as e:
        logger.error(f"Error generating AI insights: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
@main.route('/system-maintenance')
@login_required
def system_maintenance():
    """
    Display system maintenance dashboard with AI-powered predictions
    Monitors core modules without modifying their functionality
    """
    try:
        # Initialize maintenance monitor
        monitor = MaintenanceMonitor()
        
        # Get health metrics for current user
        health_metrics = monitor.check_module_health(current_user.id)
        
        # Get maintenance predictions
        maintenance_needs = monitor.predict_maintenance_needs()
        
        return render_template('system_maintenance.html',
                             health_metrics=health_metrics,
                             maintenance_needs=maintenance_needs)
        
    except Exception as e:
        logger.error(f"Error in system maintenance: {str(e)}")
        flash('Error checking system health')
        return redirect(url_for('main.dashboard'))
        
@main.route('/alerts')
@login_required
def alert_dashboard():
    """Display financial alert dashboard with configurations and active alerts"""
    try:
        # Get active alerts and configurations for the user
        active_alerts = AlertHistory.query.filter(
            AlertHistory.user_id == current_user.id,
            AlertHistory.status != 'resolved'
        ).order_by(AlertHistory.created_at.desc()).all()
        
        configurations = AlertConfiguration.query.filter_by(
            user_id=current_user.id
        ).order_by(AlertConfiguration.created_at.desc()).all()
        
        return render_template('alerts/alert_dashboard.html',
                             active_alerts=active_alerts,
                             configurations=configurations)
        
    except Exception as e:
        logger.error(f"Error loading alert dashboard: {str(e)}")
        flash('Error loading alert dashboard')
        return redirect(url_for('main.dashboard'))
        
@main.route('/alerts/create', methods=['POST'])
@login_required
def create_alert_config():
    """Create new alert configuration"""
    try:
        threshold_value = float(request.form['threshold_value'])
        if threshold_value < 0:
            raise ValueError("Threshold value must be non-negative")
        config = AlertConfiguration(
            user_id=current_user.id,
            name=request.form['name'][:100],
            alert_type=request.form['alert_type'],
            threshold_type=request.form['threshold_type'],
            threshold_value=threshold_value,
            notification_method=request.form.get('notification_method', 'web')
        )
        db.session.add(config)
        db.session.commit()
        flash('Alert configuration created successfully')
        return redirect(url_for('main.alert_dashboard'))
        
    except ValueError as ve:
        logger.error(f"Invalid alert configuration values: {str(ve)}")
        flash('Invalid configuration values')
        return redirect(url_for('main.alert_dashboard'))
    except Exception as e:
        logger.error(f"Error creating alert configuration: {str(e)}")
        db.session.rollback()
        flash('Error creating alert configuration')
        return redirect(url_for('main.alert_dashboard'))
        
@main.route('/alerts/acknowledge/<int:alert_id>', methods=['POST'])
@login_required
def acknowledge_alert(alert_id):
    """Mark an alert as acknowledged"""
    try:
        alert_system = AlertSystem()
        success = alert_system.acknowledge_alert(alert_id, current_user.id)
        
        if success:
            return jsonify({'success': True})
        return jsonify({'error': 'Alert not found or unauthorized'}), 404
        
    except Exception as e:
        logger.error(f"Error acknowledging alert: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
@main.route('/alerts/check')
@login_required
def check_alerts():
    """Check for new anomalies and generate alerts"""
    try:
        alert_system = AlertSystem()
        anomalies = alert_system.check_anomalies(current_user.id)
        
        created_alerts = []
        for anomaly in anomalies:
            alert = alert_system.create_alert(
                user_id=current_user.id,
                anomaly=anomaly,
                config_id=anomaly.get('config_id')
            )
            if alert:
                created_alerts.append(alert)
                
        return jsonify({
            'success': True,
            'alerts_created': len(created_alerts)
        })
        
    except Exception as e:
        logger.error(f"Error checking alerts: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
@main.route('/goals')
@login_required
def goals_dashboard():
    """Display financial goals dashboard"""
    try:
        goals = FinancialGoal.query.filter_by(
            user_id=current_user.id
        ).order_by(FinancialGoal.created_at.desc()).all()
        
        # Calculate overall progress statistics
        total_goals = len(goals)
        completed_goals = sum(1 for g in goals if g.status == 'completed')
        active_goals = sum(1 for g in goals if g.status == 'active')
        
        # Group goals by category
        goals_by_category = {}
        for goal in goals:
            category = goal.category or 'Uncategorized'
            if category not in goals_by_category:
                goals_by_category[category] = []
            goals_by_category[category].append(goal)
            
        return render_template('goals_dashboard.html',
                            goals=goals,
                            total_goals=total_goals,
                            completed_goals=completed_goals,
                            active_goals=active_goals,
                            goals_by_category=goals_by_category)
        
    except Exception as e:
        logger.error(f"Error loading goals dashboard: {str(e)}")
        flash('Error loading goals dashboard')
        return redirect(url_for('main.dashboard'))
        
@main.route('/goals/create', methods=['GET', 'POST'])
@login_required
def create_goal():
    """Create a new financial goal"""
    if request.method == 'POST':
        try:
            target_amount = float(request.form['target_amount'])
            current_amount = float(request.form.get('current_amount', 0))
            if target_amount <= 0:
                raise ValueError("Target amount must be positive")
            deadline = None
            if request.form.get('deadline'):
                try:
                    deadline = datetime.strptime(request.form['deadline'], '%Y-%m-%d')
                except ValueError:
                    raise ValueError("Invalid deadline date format. Use YYYY-MM-DD.")
            goal = FinancialGoal(
                user_id=current_user.id,
                name=request.form['name'][:100],
                description=request.form.get('description'),
                target_amount=target_amount,
                current_amount=current_amount,
                category=request.form.get('category'),
                deadline=deadline,
                is_recurring=bool(request.form.get('is_recurring')),
                recurrence_period=request.form.get('recurrence_period')
            )
            db.session.add(goal)
            db.session.commit()
            flash('Financial goal created successfully')
            return redirect(url_for('main.goals_dashboard'))
            
        except Exception as e:
            logger.error(f"Error creating financial goal: {str(e)}")
            db.session.rollback()
            flash('Error creating financial goal')
            return redirect(url_for('main.goals_dashboard'))
            
    return render_template('create_goal.html')
    
@main.route('/goals/<int:goal_id>/update', methods=['POST'])
@login_required
def update_goal(goal_id):
    """Update goal progress"""
    try:
        goal = FinancialGoal.query.filter_by(
            id=goal_id,
            user_id=current_user.id
        ).first_or_404()
        
        # Update goal progress
        goal.current_amount = float(request.form['current_amount'])
        
        # Update status if goal is completed
        if goal.calculate_progress() >= 100:
            goal.status = 'completed'
            
        db.session.commit()
        return jsonify({
            'success': True,
            'progress': goal.calculate_progress(),
            'status': goal.status
        })
        
    except Exception as e:
        logger.error(f"Error updating goal: {str(e)}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
        
def find_similar_transactions(description):
    """Find similar transactions based on description"""
    try:
        similar_transactions = Transaction.query.filter(
            Transaction.user_id == current_user.id,
            Transaction.description.ilike(f"%{description}%")
        ).all()
        
        return [{
            'id': t.id,
            'description': t.description,
            'explanation': t.explanation
        } for t in similar_transactions]
    except Exception as e:
        logger.error(f"Error finding similar transactions: {str(e)}")
        return []
        
@main.route('/suggest-explanation', methods=['POST'])
@login_required
def suggest_explanation_api():
    """API endpoint for ESF (Explanation Suggestion Feature)"""
    try:
        data = request.get_json()
        description = data.get('description', '').strip()
        
        if not description:
            return jsonify({'error': 'Description is required'}), 400
            
        similar_transactions = find_similar_transactions(description)
        suggestion = suggest_explanation(description, similar_transactions)
        
        return jsonify({
            'success': True,
            'suggestion': suggestion
        })
        
    except Exception as e:
        logger.error(f"Error in ESF: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
from werkzeug.utils import secure_filename
        
def find_similar_transactions(description):
    """Find similar transactions based on description"""
    try:
        similar_transactions = Transaction.query.filter(
            Transaction.user_id == current_user.id,
            Transaction.description.ilike(f"%{description}%")
        ).all()
        
        return [{
            'id': t.id,
            'description': t.description,
            'explanation': t.explanation
        } for t in similar_transactions]
    except Exception as e:
        logger.error(f"Error finding similar transactions: {str(e)}")
        return []
        
@main.route('/analyze/anomalies')
@login_required
def analyze_anomalies():
    """Analyze transactions for anomalies"""
    try:
        days_back = request.args.get('days', default=90, type=int)
        
        # Initialize anomaly detection service
        detector = AnomalyDetectionService(current_user.id)
        result = detector.detect_anomalies(days_back=days_back)
        
        if result['status'] == 'success':
            return render_template(
                'anomaly_analysis.html',
                anomalies=result['anomalies'],
                summary=result['summary'],
                analysis_period=result['analysis_period']
            )
        else:
            flash(result['message'], 'error')
            return redirect(url_for('main.dashboard'))
        
    except Exception as e:
        logger.error(f"Error in anomaly analysis route: {str(e)}")
        flash('Error analyzing anomalies', 'error')
        return redirect(url_for('main.dashboard'))
        
@main.route('/api/anomalies')
@login_required
def get_anomalies():
    """API endpoint for anomaly detection results"""
    try:
        days_back = request.args.get('days', default=90, type=int)
        
        detector = AnomalyDetectionService(current_user.id)
        result = detector.detect_anomalies(days_back=days_back)
        
        if result['status'] == 'success':
            return jsonify(result)
        else:
            return jsonify({'error': result['message']}), 400
        
    except Exception as e:
        logger.error(f"Error in anomaly API route: {str(e)}")
        return jsonify({'error': str(e)}), 500