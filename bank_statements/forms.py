"""
Forms specific to bank statement processing
"""
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SelectField, SubmitField
from wtforms.validators import DataRequired
from flask_login import current_user
from models import Account

class BankStatementUploadForm(FlaskForm):
    """Form for bank statement upload with proper validation"""
    account = SelectField(
        'Bank Account',
        validators=[DataRequired(message="Please select a bank account")],
        description='Select the bank account this statement belongs to'
    )

    file = FileField(
        'Bank Statement File',
        validators=[
            FileRequired(message="Please select a file to upload"),
            FileAllowed(
                ['csv', 'xlsx'],
                'Only CSV (.csv) and Excel (.xlsx) files can be imported here. '
                'Save an older .xls file as .xlsx first; PDF statements go '
                'through Import PDF Statement.')
        ],
        description='Upload bank statement in CSV or Excel format'
    )

    submit = SubmitField('Upload Statement')

    def __init__(self, *args, **kwargs):
        """Initialize form and populate account choices"""
        super(BankStatementUploadForm, self).__init__(*args, **kwargs)
        if current_user and current_user.is_authenticated:
            try:
                # Get bank accounts (starting with ca.810)
                bank_accounts = Account.query.filter(
                    Account.user_id == current_user.id,
                    Account.link.like('ca.810%')
                ).all()

                self.account.choices = [
                    (str(acc.id), f"{acc.link} - {acc.name}")
                    for acc in bank_accounts
                ]
            except Exception as e:
                self.account.choices = []
                # Log the error but don't raise it to allow form rendering
                print(f"Error loading bank accounts: {str(e)}")