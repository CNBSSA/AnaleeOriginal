"""User and related models for the application"""
from typing import Dict, List, Optional
import logging
import os
from datetime import datetime, timedelta
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, Text, DateTime, Enum as SQLEnum, Index
from sqlalchemy.orm import relationship
from werkzeug.security import generate_password_hash, check_password_hash

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database
db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model with enhanced security features"""
    __tablename__ = 'user'

    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(256))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_admin = Column(Boolean, default=False)
    is_deleted = Column(Boolean, default=False)
    subscription_status = Column(String(20), default='active')  # Changed default to active

    # Define relationships with explicit back_populates
    financial_goals = relationship('FinancialGoal', back_populates='user', cascade='all, delete-orphan')
    transactions = relationship('Transaction', back_populates='user', cascade='all, delete-orphan')
    accounts = relationship('Account', back_populates='user', cascade='all, delete-orphan')
    company_settings = relationship('CompanySettings', back_populates='user', uselist=False, cascade='all, delete-orphan')
    bank_statement_uploads = relationship('BankStatementUpload', back_populates='user', cascade='all, delete-orphan')
    alert_configurations = relationship('AlertConfiguration', back_populates='user', cascade='all, delete-orphan')
    alert_history = relationship('AlertHistory', back_populates='user', cascade='all, delete-orphan')
    historical_data = relationship('HistoricalData', back_populates='user', cascade='all, delete-orphan')
    risk_assessments = relationship('RiskAssessment', back_populates='user', cascade='all, delete-orphan')
    financial_recommendations = relationship('FinancialRecommendation', back_populates='user', cascade='all, delete-orphan')


    def set_password(self, password: str) -> None:
        """Set hashed password"""
        if not password:
            raise ValueError('Password cannot be empty')
        try:
            self.password_hash = generate_password_hash(password)
            logger.info(f"Password set successfully for user {self.id}")
        except Exception as e:
            logger.error(f"Error setting password for user {self.id}: {str(e)}")
            raise ValueError(f"Error setting password: {str(e)}")

    def check_password(self, password: str) -> bool:
        """Check if provided password matches hash"""
        if not password or not self.password_hash:
            return False
        try:
            return check_password_hash(self.password_hash, password)
        except Exception as e:
            logger.error(f"Error checking password for user {self.id}: {str(e)}")
            return False

    @property
    def is_active(self) -> bool:
        """Check if user is active for Flask-Login"""
        return not self.is_deleted and self.subscription_status in ['active', 'pending']

    def __repr__(self):
        return f'<User {self.username}>'

class FinancialGoal(db.Model):
    """Model for financial goals"""
    __tablename__ = 'financial_goal'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    target_amount = Column(Float, nullable=False)
    current_amount = Column(Float, default=0.0)
    start_date = Column(DateTime, default=datetime.utcnow)
    deadline = Column(DateTime)
    category = Column(String(50))
    status = Column(String(20), default='active')
    is_recurring = Column(Boolean, default=False)
    recurrence_period = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User", back_populates="financial_goals")

    def calculate_progress(self):
        """Calculate current progress as percentage"""
        if self.target_amount == 0:
            return 0
        return min(100, (self.current_amount / self.target_amount) * 100)

    def get_status_details(self):
        """Get detailed status information"""
        progress = self.calculate_progress()
        days_remaining = None
        if self.deadline:
            days_remaining = (self.deadline - datetime.utcnow()).days

        return {
            'progress': progress,
            'days_remaining': days_remaining,
            'is_overdue': days_remaining < 0 if days_remaining is not None else False,
            'status': self.status
        }

    def __repr__(self):
        return f'<FinancialGoal {self.name}: {self.current_amount}/{self.target_amount}>'

class Transaction(db.Model):
    """Model for financial transactions"""
    __tablename__ = 'transaction'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    description = Column(String(200), nullable=False)
    amount = Column(Float, nullable=False)
    category = Column(String(50))
    user_id = Column(Integer, ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    account_id = Column(Integer, ForeignKey('account.id', ondelete='CASCADE'))
    file_id = Column(Integer, ForeignKey('uploaded_file.id', ondelete='SET NULL'))
    explanation = Column(String(500))
    # AI-generated fields used by the iCountant interface
    ai_category = Column(String(50))
    ai_confidence = Column(Float)
    ai_explanation = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('ix_transaction_user_date', 'user_id', 'date'),
        Index('ix_transaction_user_account', 'user_id', 'account_id'),
        Index('ix_transaction_file', 'file_id'),
    )

    # Define relationships with back_populates
    user = relationship('User', back_populates='transactions')
    account = relationship('Account', back_populates='transactions')
    file = relationship('UploadedFile', back_populates='transactions')

    def __repr__(self):
        return f'<Transaction {self.date}: {self.description}>'

class Account(db.Model):
    """Model for financial accounts"""
    __tablename__ = 'account'

    id = Column(Integer, primary_key=True)
    link = Column(String(20), nullable=False)
    category = Column(String(100), nullable=False)
    sub_category = Column(String(100))
    account_code = Column(String(20))
    name = Column(String(100), nullable=False)
    user_id = Column(Integer, ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('ix_account_user_link', 'user_id', 'link', unique=True),
        Index('ix_account_user_active', 'user_id', 'is_active'),
    )

    # Define relationships with back_populates
    user = relationship('User', back_populates='accounts')
    transactions = relationship('Transaction', back_populates='account', cascade='all, delete-orphan')
    bank_statement_uploads = relationship('BankStatementUpload', back_populates='account', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Account {self.link}: {self.name}>'

class BankStatementUpload(db.Model):
    """Model for tracking bank statement uploads"""
    __tablename__ = 'bank_statement_upload'

    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    upload_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    status = Column(
        SQLEnum('pending', 'processing', 'completed', 'failed', 
               name='bank_statement_status', 
               create_constraint=True),
        nullable=False,
        default='pending'
    )
    error_message = Column(Text)
    processing_notes = Column(Text)
    account_id = Column(Integer, ForeignKey('account.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(Integer, ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    user = relationship("User", back_populates="bank_statement_uploads")
    account = relationship('Account', back_populates='bank_statement_uploads')

    def __repr__(self):
        return f'<BankStatementUpload {self.filename} ({self.status})>'

    @property
    def is_processed(self):
        """Check if upload has been processed"""
        return self.status in ('completed', 'failed')

    def set_error(self, message: str):
        """Set error status with message"""
        self.status = 'failed'
        self.error_message = message

    def set_success(self, notes: Optional[str] = None):
        """Mark upload as successfully processed"""
        self.status = 'completed'
        if notes:
            self.processing_notes = notes

class CompanySettings(db.Model):
    __tablename__ = 'company_settings'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    company_name = Column(String(200), nullable=False)
    registration_number = Column(String(50))
    tax_number = Column(String(50))
    vat_number = Column(String(50))
    address = Column(Text)
    financial_year_end = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User", back_populates="company_settings")
    
    def get_financial_year(self, date=None, year=None):
        if date is None:
            date = datetime.utcnow()
        
        if year is not None:
            start_year = year
        else:
            if date.month > self.financial_year_end:
                start_year = date.year
            else:
                start_year = date.year - 1
        
        end_year = start_year + 1
        
        if self.financial_year_end == 12:
            start_date = datetime(start_year + 1, 1, 1)
        else:
            start_date = datetime(start_year, self.financial_year_end + 1, 1)
        
        if self.financial_year_end == 12:
            end_date = datetime(end_year, 12, 31)
        else:
            if self.financial_year_end == 2:
                last_day = 29 if end_year % 4 == 0 and (end_year % 100 != 0 or end_year % 400 == 0) else 28
            elif self.financial_year_end in [4, 6, 9, 11]:
                last_day = 30
            else:
                last_day = 31
            
            end_date = datetime(end_year, self.financial_year_end, last_day)
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'start_year': start_year,
            'end_year': end_year
        }

    def __repr__(self):
        return f'<CompanySettings {self.company_name}>'

class UploadedFile(db.Model):
    __tablename__ = 'uploaded_file'

    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    transactions = relationship('Transaction', back_populates='file')

class KeywordRule(db.Model):
    __tablename__ = 'keyword_rule'

    id = Column(Integer, primary_key=True)
    keyword = Column(String(200), nullable=False)
    category = Column(String(100), nullable=False)
    priority = Column(Integer, default=1)
    is_regex = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<KeywordRule {self.keyword}: {self.category}>'

class HistoricalData(db.Model):
    __tablename__ = 'historical_data'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    description = Column(String(200), nullable=False)
    amount = Column(Float, nullable=False)
    explanation = Column(String(200))
    account_id = Column(Integer, ForeignKey('account.id'))
    user_id = Column(Integer, ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    account = relationship('Account')
    user = relationship("User", back_populates="historical_data")

    def __repr__(self):
        return f'<HistoricalData {self.date}: {self.description}>'

class RiskAssessment(db.Model):
    __tablename__ = 'risk_assessment'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    assessment_date = Column(DateTime, default=datetime.utcnow)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)  # low, medium, high
    assessment_type = Column(String(50), nullable=False)  # liquidity, solvency, operational
    findings = Column(Text)
    recommendations = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User", back_populates="risk_assessments")

    def __repr__(self):
        return f'<RiskAssessment {self.assessment_date}: {self.risk_level}>'

class RiskIndicator(db.Model):
    __tablename__ = 'risk_indicator'

    id = Column(Integer, primary_key=True)
    assessment_id = Column(Integer, ForeignKey('risk_assessment.id', ondelete='CASCADE'), nullable=False)
    indicator_name = Column(String(100), nullable=False)
    indicator_value = Column(Float, nullable=False)
    threshold_value = Column(Float, nullable=False)
    is_breach = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    assessment = relationship('RiskAssessment')

    def __repr__(self):
        return f'<RiskIndicator {self.indicator_name}: {self.indicator_value}>'

class AlertConfiguration(db.Model):
    __tablename__ = 'alert_configuration'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    name = Column(String(100), nullable=False)
    alert_type = Column(String(50), nullable=False)  # transaction, balance, pattern
    threshold_type = Column(String(50), nullable=False)  # amount, percentage, count
    threshold_value = Column(Float, nullable=False)
    is_active = Column(Boolean, default=True)
    notification_method = Column(String(50), default='web')  # web, email
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User", back_populates="alert_configurations")

    def __repr__(self):
        return f'<AlertConfiguration {self.name}: {self.alert_type}>'

class AlertHistory(db.Model):
    __tablename__ = 'alert_history'

    id = Column(Integer, primary_key=True)
    alert_config_id = Column(Integer, ForeignKey('alert_configuration.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(Integer, ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    alert_message = Column(Text, nullable=False)
    severity = Column(String(20), nullable=False)  # low, medium, high
    status = Column(String(20), default='new')  # new, acknowledged, resolved
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    alert_config = relationship('AlertConfiguration')
    user = relationship("User", back_populates="alert_history")

    def __repr__(self):
        return f'<AlertHistory {self.severity}: {self.alert_message[:50]}>'

class FinancialRecommendation(db.Model):
    __tablename__ = 'financial_recommendation'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    category = Column(String(50), nullable=False)  # cashflow, investment, cost_reduction, etc.
    priority = Column(String(20), nullable=False)  # high, medium, low
    recommendation = Column(Text, nullable=False)
    impact_score = Column(Float)  # Estimated financial impact
    status = Column(String(20), default='new')  # new, in_progress, completed, dismissed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    applied_at = Column(DateTime)
    user = relationship("User", back_populates="financial_recommendations")

    def __repr__(self):
        return f'<FinancialRecommendation {self.category}: {self.recommendation[:50]}>'

class RecommendationMetrics(db.Model):
    __tablename__ = 'recommendation_metrics'

    id = Column(Integer, primary_key=True)
    recommendation_id = Column(Integer, ForeignKey('financial_recommendation.id', ondelete='CASCADE'), nullable=False)
    metric_name = Column(String(100), nullable=False)
    baseline_value = Column(Float)
    current_value = Column(Float)
    target_value = Column(Float)
    measured_at = Column(DateTime, default=datetime.utcnow)
    recommendation = relationship('FinancialRecommendation')

    def __repr__(self):
        return f'<RecommendationMetrics {self.metric_name}: {self.current_value}>'

class AdminChartOfAccounts(db.Model):
    __tablename__ = 'admin_chart_of_accounts'

    id = Column(Integer, primary_key=True)
    link = Column(String(20), nullable=False, unique=True)  # Maps to 'Links' column
    code = Column(String(20), nullable=False)  # Maps to 'Code' column
    name = Column(String(100), nullable=False)  # Maps to 'Account Name' column
    category = Column(String(50), nullable=False)  # Maps to 'Category' column
    sub_category = Column(String(50))  # Maps to 'Sub Category' column
    description = Column(Text)  # Additional field for account details
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<AdminChartOfAccounts {self.code}: {self.name}>'

    @property
    def serialize(self):
        """Return object data in easily serializable format"""
        return {
            'id': self.id,
            'link': self.link,
            'code': self.code,
            'name': self.name,
            'category': self.category,
            'sub_category': self.sub_category,
            'description': self.description
        }