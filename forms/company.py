"""
Forms for company settings management
"""
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SelectField, SubmitField
from wtforms.validators import DataRequired, Length, Optional

class CompanySettingsForm(FlaskForm):
    """Form for company settings with CSRF protection"""
    company_name = StringField('Company Name', 
                           validators=[DataRequired(), Length(max=100)])
    registration_number = StringField('Registration Number', 
                                  validators=[Optional(), Length(max=50)])
    tax_number = StringField('Tax Number', 
                         validators=[Optional(), Length(max=50)])
    vat_number = StringField('VAT Number', 
                         validators=[Optional(), Length(max=50)])
    address = TextAreaField('Address', 
                        validators=[Optional(), Length(max=200)])
    financial_year_end = SelectField('Financial Year End',
                                 choices=[(str(i), str(i)) for i in range(1, 13)],
                                 validators=[DataRequired()])
    entity_id = SelectField(
        'Entity Type',
        coerce=int,
        validators=[DataRequired()],
    )
    submit = SubmitField('Save Settings')
