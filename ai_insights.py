import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
import anthropic
from nlp_utils import get_claude_client as get_openai_client, categorize_transaction

# Enhanced logging configuration
logger = logging.getLogger(__name__)

class AIServiceStatus:
    """Track AI service status and errors"""
    def __init__(self):
        self.last_error = None
        self.error_count = 0
        self.last_success = None
        self.consecutive_failures = 0

    def record_error(self, error: Exception) -> None:
        """Record an error occurrence"""
        self.last_error = {
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'message': str(error)
        }
        self.error_count += 1
        self.consecutive_failures += 1

    def record_success(self) -> None:
        """Record a successful operation"""
        self.last_success = datetime.now()
        self.consecutive_failures = 0

class FinancialInsightsGenerator:
    """
    Class responsible for generating financial insights using AI.
    Handles both OpenAI-powered and fallback basic analysis.
    """
    def __init__(self):
        self.client = get_openai_client()
        self.env = os.environ.get('FLASK_ENV', 'development')
        self.service_status = AIServiceStatus()

    def _log_service_status(self, operation: str) -> None:
        """Log current service status"""
        status_info = {
            'operation': operation,
            'error_count': self.service_status.error_count,
            'consecutive_failures': self.service_status.consecutive_failures,
            'last_error': self.service_status.last_error,
            'last_success': self.service_status.last_success
        }
        logger.info(f"AI Service Status: {status_info}")

    def generate_transaction_insights(self, transaction_data: List[Dict]) -> Dict:
        """
        Generate AI-powered insights for transactions with enhanced error handling.

        Args:
            transaction_data (List[Dict]): List of transaction information

        Returns:
            Dict: Generated insights and analysis results with error status
        """
        try:
            if not isinstance(transaction_data, list) or not transaction_data:
                logger.error("Invalid or empty transaction data provided")
                return self._generate_fallback_insights([], error="No valid transaction data provided")

            # For single transaction analysis, use the first transaction
            transaction = transaction_data[0]

            if not self.client:
                logger.warning("AI service client unavailable, using fallback analysis")
                return self._generate_fallback_insights([transaction], error="AI service temporarily unavailable")

            # Prepare transaction for analysis
            transaction_summary = self._prepare_transaction_summary([transaction])

            # Get AI categorization with error handling
            try:
                category, confidence, explanation = categorize_transaction(transaction.get('description', ''))
            except (APIError, RateLimitError) as e:
                logger.error(f"AI API Error in categorization: {str(e)}")
                self.service_status.record_error(e)
                return self._generate_fallback_insights([transaction], error=f"AI service error: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error in categorization: {str(e)}")
                self.service_status.record_error(e)
                category, confidence, explanation = None, 0, "Categorization unavailable"

            # Generate insights using OpenAI with enhanced error handling
            try:
                max_tokens = 300 if self.env == 'production' else 500
                temperature = 0.5 if self.env == 'production' else 0.7

                response = self.client.messages.create(
                    model="claude-opus-4-7",
                    max_tokens=max_tokens,
                    system="You are a financial analyst providing insights on transaction data.",
                    messages=[{"role": "user", "content": f"Analyze this financial transaction and provide key insights: {transaction_summary}"}]
                )

                insights = response.content[0].text
                self.service_status.record_success()
                self._log_service_status("generate_insights")

            except anthropic.APIError as e:
                logger.error(f"Claude API Error: {str(e)}")
                self.service_status.record_error(e)
                return self._generate_fallback_insights([transaction], error=f"AI service error: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error generating insights: {str(e)}")
                self.service_status.record_error(e)
                insights = "AI insights currently unavailable. Using basic analysis."

            return {
                'success': True,
                'insights': insights,
                'generated_at': datetime.now().isoformat(),
                'analysis_type': 'ai_powered',
                'service_status': {
                    'error_count': self.service_status.error_count,
                    'last_success': self.service_status.last_success.isoformat() if self.service_status.last_success else None
                },
                'category_suggestion': {
                    'category': category,
                    'confidence': confidence,
                    'explanation': explanation
                }
            }

        except Exception as e:
            logger.error(f"Critical error in insight generation: {str(e)}")
            self.service_status.record_error(e)
            return self._generate_fallback_insights(
                [transaction_data[0]] if transaction_data else [], 
                error=f"Critical error: {str(e)}"
            )

    def generate_insights(self, transactions: List[Dict]) -> Dict:
        """Generate insights from transaction data using AI."""
        try:
            if not self.client:
                return self._generate_fallback_insights(transactions)

            # Prepare transaction data for analysis
            transaction_summary = self._prepare_transaction_summary(transactions)

            # Get AI categorization for the latest transaction
            latest_transaction = transactions[0] if transactions else None
            if latest_transaction:
                try:
                    category, confidence, explanation = categorize_transaction(latest_transaction['description'])
                except (APIError, RateLimitError) as e:
                    logger.error(f"AI API Error in categorization: {str(e)}")
                    self.service_status.record_error(e)
                    return self._generate_fallback_insights(transactions, error=f"AI service error during categorization: {str(e)}")
                except Exception as e:
                    logger.error(f"Unexpected error in categorization: {str(e)}")
                    self.service_status.record_error(e)
                    category, confidence, explanation = None, 0, "Categorization unavailable"
            else:
                category, confidence, explanation = None, 0, "No transaction to analyze"

            # Generate combined insights using OpenAI with environment-specific handling
            try:
                if self.env == 'production':
                    # Production: More conservative token usage and caching
                    max_tokens = 300
                    temperature = 0.5
                else:
                    # Development: More experimental
                    max_tokens = 500
                    temperature = 0.7

                response = self.client.messages.create(
                    model="claude-opus-4-7",
                    max_tokens=max_tokens,
                    system="You are a financial analyst providing insights on transaction data.",
                    messages=[{"role": "user", "content": f"Analyze these financial transactions and provide key insights: {transaction_summary}"}]
                )

                insights = response.content[0].text
                self.service_status.record_success()
                self._log_service_status("generate_insights")

            except anthropic.APIError as e:
                logger.error(f"Claude API Error: {str(e)}")
                self.service_status.record_error(e)
                return self._generate_fallback_insights(transactions, error=f"AI service error: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error generating insights: {str(e)}")
                self.service_status.record_error(e)
                insights = "AI insights currently unavailable. Using basic analysis."

            return {
                'success': True,
                'insights': insights,
                'generated_at': datetime.now().isoformat(),
                'analysis_type': 'ai_powered',
                'service_status': {
                    'error_count': self.service_status.error_count,
                    'last_success': self.service_status.last_success.isoformat() if self.service_status.last_success else None
                },
                'category_suggestion': {
                    'category': category,
                    'confidence': confidence,
                    'explanation': explanation
                }
            }

        except Exception as e:
            logger.error(f"Critical error generating insights: {str(e)}")
            self.service_status.record_error(e)
            return self._generate_fallback_insights(transactions, error=f"Critical error: {str(e)}")

    def _prepare_transaction_summary(self, transactions: List[Dict]) -> str:
        """Prepare transaction data for AI analysis."""
        if not transactions:
            return "No transactions to analyze"

        total_income = sum(t['amount'] for t in transactions if t['amount'] > 0)
        total_expenses = sum(abs(t['amount']) for t in transactions if t['amount'] < 0)
        transaction_count = len(transactions)

        # Categorize transactions
        categories = {}
        for t in transactions:
            category = t.get('category', 'Uncategorized')
            if category not in categories:
                categories[category] = {'count': 0, 'total': 0}
            categories[category]['count'] += 1
            categories[category]['total'] += abs(t['amount'])

        summary = (
            f"Financial Summary:\n"
            f"Total Income: ${total_income:,.2f}\n"
            f"Total Expenses: ${total_expenses:,.2f}\n"
            f"Transaction Count: {transaction_count}\n\n"
            f"Category Breakdown:\n"
        )

        for category, data in categories.items():
            summary += f"- {category}: ${data['total']:,.2f} ({data['count']} transactions)\n"

        # Add latest transaction details for better context
        if transactions:
            latest = transactions[0]
            summary += f"\nLatest Transaction:\n"
            summary += f"Description: {latest.get('description', 'N/A')}\n"
            summary += f"Amount: ${abs(latest.get('amount', 0)):,.2f} "
            summary += "credit" if latest.get('amount', 0) < 0 else "debit"

        return summary

    def _generate_fallback_insights(self, transactions: List[Dict], error: Optional[str] = None) -> Dict:
        """Generate basic insights without AI when service is unavailable."""
        try:
            if not transactions or not isinstance(transactions, list):
                return {
                    'success': True,
                    'insights': "No transactions to analyze",
                    'generated_at': datetime.now().isoformat(),
                    'analysis_type': 'basic',
                    'error': error,
                    'service_status': {
                        'error_count': self.service_status.error_count,
                        'last_error': self.service_status.last_error
                    },
                    'category_suggestion': {
                        'category': None,
                        'confidence': 0,
                        'explanation': "No transaction to analyze"
                    }
                }

            # Process valid transaction data
            if isinstance(transactions[0], dict):
                latest = transactions[0]
                amount = latest.get('amount', 0)

                total_income = sum(t.get('amount', 0) for t in transactions if t.get('amount', 0) > 0)
                total_expenses = sum(abs(t.get('amount', 0)) for t in transactions if t.get('amount', 0) < 0)
                net_change = total_income - total_expenses

                category_guess = "expense" if amount < 0 else "income"

                insights = (
                    f"Basic Financial Analysis:\n"
                    f"- Total Income: ${total_income:,.2f}\n"
                    f"- Total Expenses: ${total_expenses:,.2f}\n"
                    f"- Net Change: ${net_change:,.2f}\n\n"
                    f"Transaction appears to be an {category_guess}.\n"
                    f"This is a basic analysis generated without AI assistance."
                )

                if error:
                    insights += f"\n\nNote: {error}"

                return {
                    'success': True,
                    'insights': insights,
                    'generated_at': datetime.now().isoformat(),
                    'analysis_type': 'basic',
                    'error': error,
                    'service_status': {
                        'error_count': self.service_status.error_count,
                        'last_error': self.service_status.last_error
                    },
                    'category_suggestion': {
                        'category': category_guess,
                        'confidence': 0.5,
                        'explanation': f"Basic categorization based on amount being {'negative' if amount < 0 else 'positive'}"
                    }
                }
            else:
                logger.error("Invalid transaction data format")
                return {
                    'success': False,
                    'error': "Invalid transaction data format",
                    'analysis_type': 'error',
                    'service_status': {
                        'error_count': self.service_status.error_count,
                        'last_error': self.service_status.last_error
                    }
                }

        except Exception as e:
            logger.error(f"Critical error in fallback analysis: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'generated_at': datetime.now().isoformat(),
                'analysis_type': 'error',
                'service_status': {
                    'error_count': self.service_status.error_count,
                    'last_error': self.service_status.last_error
                },
                'category_suggestion': {
                    'category': None,
                    'confidence': 0,
                    'explanation': f"Error in fallback analysis: {str(e)}"
                }
            }