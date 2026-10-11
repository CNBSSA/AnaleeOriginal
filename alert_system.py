"""
Financial Alert System Module
Handles anomaly detection and alert generation without modifying core functionalities
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy import func
from models import db, AlertConfiguration, AlertHistory, Transaction, Account

# Configure logging
logger = logging.getLogger(__name__)

class AlertSystem:
    """
    Handles financial anomaly detection and alert management
    Maintains isolation from core system components
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def check_anomalies(self, user_id: int) -> List[Dict]:
        """
        Check for financial anomalies based on user configurations
        
        Args:
            user_id: ID of the user to check anomalies for
            
        Returns:
            List of detected anomalies
        """
        try:
            # Get active alert configurations
            configurations = AlertConfiguration.query.filter_by(
                user_id=user_id,
                is_active=True
            ).all()
            
            detected_anomalies = []
            
            for config in configurations:
                anomalies = self._process_alert_configuration(config)
                if anomalies:
                    detected_anomalies.extend(anomalies)
                    
            return detected_anomalies
            
        except Exception as e:
            self.logger.error(f"Error checking anomalies: {str(e)}")
            return []

    def _process_alert_configuration(self, config: AlertConfiguration) -> List[Dict]:
        """
        Process individual alert configuration
        
        Args:
            config: AlertConfiguration object to process
            
        Returns:
            List of anomalies detected for this configuration
        """
        try:
            if config.alert_type == 'transaction':
                return self._check_transaction_anomalies(config)
            elif config.alert_type == 'balance':
                return self._check_balance_anomalies(config)
            elif config.alert_type == 'pattern':
                return self._check_pattern_anomalies(config)
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error processing alert configuration {config.id}: {str(e)}")
            return []

    def _check_transaction_anomalies(self, config: AlertConfiguration) -> List[Dict]:
        """Check for transaction-based anomalies"""
        try:
            recent_transactions = Transaction.query.filter(
                Transaction.user_id == config.user_id,
                Transaction.date >= datetime.utcnow() - timedelta(days=30)
            ).all()
            
            anomalies = []
            for transaction in recent_transactions:
                if config.threshold_type == 'amount' and abs(transaction.amount) > config.threshold_value:
                    anomalies.append({
                        'type': 'transaction',
                        'severity': 'high' if abs(transaction.amount) > config.threshold_value * 1.5 else 'medium',
                        'message': f'Large transaction detected: R{abs(transaction.amount):,.2f}',
                        'transaction_id': transaction.id
                    })
                    
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error checking transaction anomalies: {str(e)}")
            return []

    def _check_balance_anomalies(self, config: AlertConfiguration) -> List[Dict]:
        """Check for balance-based anomalies"""
        try:
            # Get account balances
            accounts = Account.query.filter_by(user_id=config.user_id).all()
            
            anomalies = []
            for account in accounts:
                balance = sum(t.amount for t in account.transactions)
                
                if config.threshold_type == 'amount' and abs(balance) > config.threshold_value:
                    anomalies.append({
                        'type': 'balance',
                        'severity': 'high' if abs(balance) > config.threshold_value * 1.5 else 'medium',
                        'message': f'Account balance threshold exceeded: {account.name}',
                        'account_id': account.id
                    })
                    
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error checking balance anomalies: {str(e)}")
            return []

    def _check_pattern_anomalies(self, config: AlertConfiguration) -> List[Dict]:
        """Check for pattern-based anomalies"""
        try:
            # Get recent transactions for pattern analysis
            recent_transactions = Transaction.query.filter(
                Transaction.user_id == config.user_id,
                Transaction.date >= datetime.utcnow() - timedelta(days=90)
            ).order_by(Transaction.date).all()
            
            anomalies = []
            if len(recent_transactions) >= 3:
                # Check for unusual patterns
                for i in range(2, len(recent_transactions)):
                    current = recent_transactions[i].amount
                    prev1 = recent_transactions[i-1].amount
                    prev2 = recent_transactions[i-2].amount
                    
                    # Detect sudden changes in transaction patterns
                    if abs(current) > abs(prev1) * 3 and abs(current) > abs(prev2) * 3:
                        anomalies.append({
                            'type': 'pattern',
                            'severity': 'medium',
                            'message': 'Unusual transaction pattern detected',
                            'transaction_id': recent_transactions[i].id
                        })
                        
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error checking pattern anomalies: {str(e)}")
            return []

    def create_alert(self, user_id: int, anomaly: Dict, config_id: int) -> Optional[AlertHistory]:
        """
        Create alert history entry for detected anomaly
        
        Args:
            user_id: ID of the user
            anomaly: Detected anomaly information
            config_id: ID of the alert configuration
            
        Returns:
            Created AlertHistory object or None if error
        """
        try:
            alert = AlertHistory(
                alert_config_id=config_id,
                user_id=user_id,
                alert_message=anomaly['message'],
                severity=anomaly['severity']
            )
            
            db.session.add(alert)
            db.session.commit()
            
            return alert
            
        except Exception as e:
            self.logger.error(f"Error creating alert: {str(e)}")
            db.session.rollback()
            return None

    def get_active_alerts(self, user_id: int) -> List[AlertHistory]:
        """Get list of active alerts for user"""
        try:
            return AlertHistory.query.filter(
                AlertHistory.user_id == user_id,
                AlertHistory.status != 'resolved'
            ).order_by(AlertHistory.created_at.desc()).all()
            
        except Exception as e:
            self.logger.error(f"Error getting active alerts: {str(e)}")
            return []

    def acknowledge_alert(self, alert_id: int, user_id: int) -> bool:
        """Mark alert as acknowledged"""
        try:
            alert = AlertHistory.query.filter_by(
                id=alert_id,
                user_id=user_id
            ).first()
            
            if alert:
                alert.status = 'acknowledged'
                alert.updated_at = datetime.utcnow()
                db.session.commit()
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {str(e)}")
            db.session.rollback()
            return False
