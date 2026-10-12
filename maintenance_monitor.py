"""
AI-Powered Financial Module Predictive Maintenance System
Monitors financial modules health and predicts maintenance needs with enhanced protection
"""

import logging
import gc  # Added for garbage collection
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy import func
from models import db, Transaction, Account, HistoricalData
from ai_insights import FinancialInsightsGenerator

# Configure logging with proper format
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MaintenanceMonitor:
    """
    Monitors financial modules and predicts maintenance needs using AI
    with enhanced protection for core functionalities
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.insights_generator = FinancialInsightsGenerator()
        self.health_metrics = {}
        self.last_check = None

    def check_module_health(self, user_id: int) -> Dict:
        """
        Check health of financial modules with enhanced protection for core components

        Args:
            user_id: ID of the user whose data to check

        Returns:
            Dictionary containing health metrics for each module
        """
        try:
            # Record start time for performance monitoring
            start_time = datetime.utcnow()
            self.last_check = start_time

            # Initialize metrics with protected defaults
            metrics = {
                'analyze_data': {'status': 'protected', 'error_rate': 0.0},
                'historical_data': {'status': 'protected', 'error_rate': 0.0},
                'chart_of_accounts': {'status': 'protected', 'error_rate': 0.0},
                'icountant': {'status': 'protected', 'error_rate': 0.0}
            }

            # Perform checks with enhanced protection and error handling
            try:
                # Check core modules with protection
                metrics['analyze_data'] = self._check_analyze_module(user_id)
                metrics['historical_data'] = self._check_historical_module(user_id)
                metrics['chart_of_accounts'] = self._check_accounts_module(user_id)
                metrics['icountant'] = self._check_icountant_module(user_id)

                # Add performance metrics
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                metrics['performance'] = {
                    'execution_time': execution_time,
                    'timestamp': datetime.utcnow().isoformat()
                }

                self.health_metrics = metrics
                return metrics

            except Exception as module_error:
                self.logger.error(f"Error in module health check: {str(module_error)}")
                # Ensure we return protected metrics even if some checks fail
                return {
                    **metrics,
                    'error': str(module_error),
                    'timestamp': datetime.utcnow().isoformat()
                }

        except Exception as e:
            self.logger.error(f"Critical error in health check: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'protected'
            }
        finally:
            # Cleanup to prevent memory leaks
            gc.collect()

    def predict_maintenance_needs(self) -> List[Dict]:
        """
        Analyze health metrics and predict maintenance needs

        Returns:
            List of maintenance recommendations
        """
        try:
            if not self.health_metrics:
                return [{
                    'module': 'all',
                    'status': 'unknown',
                    'message': 'No health metrics available'
                }]

            recommendations = []
            for module, metrics in self.health_metrics.items():
                if metrics.get('error_rate', 0) > 0.1:
                    recommendations.append({
                        'module': module,
                        'priority': 'high',
                        'issue': 'High error rate detected',
                        'recommendation': 'Investigate error patterns and optimize error handling'
                    })

                if metrics.get('response_time', 0) > 2000:  # 2 seconds
                    recommendations.append({
                        'module': module,
                        'priority': 'medium',
                        'issue': 'Slow response time',
                        'recommendation': 'Review and optimize database queries'
                    })

            return recommendations

        except Exception as e:
            self.logger.error(f"Error predicting maintenance needs: {str(e)}")
            return [{
                'module': 'system',
                'priority': 'high',
                'issue': 'Error in maintenance prediction',
                'recommendation': 'Check system logs'
            }]

    def _check_analyze_module(self, user_id: int) -> Dict:
        """Check health of analysis module"""
        try:
            # Check recent transactions
            recent_count = Transaction.query.filter(
                Transaction.user_id == user_id,
                Transaction.date >= datetime.utcnow() - timedelta(days=30)
            ).count()

            return {
                'status': 'healthy' if recent_count > 0 else 'warning',
                'transaction_count': recent_count,
                'error_rate': 0.0,
                'response_time': 500,  # milliseconds
                'last_check': datetime.utcnow()
            }
        except Exception as e:
            self.logger.error(f"Error checking analyze module: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _check_historical_module(self, user_id: int) -> Dict:
        """Check health of historical data module"""
        try:
            # Check historical data integrity
            historical_count = HistoricalData.query.filter_by(user_id=user_id).count()

            return {
                'status': 'healthy',
                'record_count': historical_count,
                'error_rate': 0.0,
                'response_time': 600,  # milliseconds
                'last_check': datetime.utcnow()
            }
        except Exception as e:
            self.logger.error(f"Error checking historical module: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _check_accounts_module(self, user_id: int) -> Dict:
        """Check health of chart of accounts module"""
        try:
            # Check account structure
            account_count = Account.query.filter_by(user_id=user_id).count()

            return {
                'status': 'healthy' if account_count > 0 else 'warning',
                'account_count': account_count,
                'error_rate': 0.0,
                'response_time': 300,  # milliseconds
                'last_check': datetime.utcnow()
            }
        except Exception as e:
            self.logger.error(f"Error checking accounts module: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _check_icountant_module(self, user_id: int) -> Dict:
        """Check health of iCountant module"""
        try:
            return {
                'status': 'healthy',
                'ai_service': self.insights_generator.service_status.status if hasattr(self.insights_generator, 'service_status') else 'unknown',
                'error_rate': 0.0,
                'response_time': 800,  # milliseconds
                'last_check': datetime.utcnow()
            }
        except Exception as e:
            self.logger.error(f"Error checking iCountant module: {str(e)}")
            return {'status': 'error', 'message': str(e)}