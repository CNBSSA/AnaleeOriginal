import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from nlp_utils import get_claude_client as get_openai_client, categorize_transaction
from utils.hybrid_predictor import HybridPredictor
from models import HistoricalData, db
from flask import current_app

logger = logging.getLogger(__name__)

class HistoricalDataAI:
    def __init__(self):
        self.client = get_openai_client()
        self.hybrid_predictor = HybridPredictor()
        self.env = current_app.config.get('ENV', 'development')

    def suggest_missing_details(self, entry: Dict) -> Dict[str, str]:
        """Generate suggestions for missing transaction details."""
        try:
            suggestions = {}

            # Check which fields need suggestions
            if not entry.get('explanation') and entry.get('description'):
                suggestions['explanation'] = self._suggest_explanation(entry['description'])

            if not entry.get('account') and entry.get('description'):
                suggestions['account'] = self._suggest_account(
                    entry['description'],
                    entry.get('explanation', ''),
                    entry.get('amount', 0)
                )

            return suggestions

        except Exception as e:
            logger.error(f"Error generating suggestions: {str(e)}")
            return {}

    def _suggest_explanation(self, description: str) -> Optional[str]:
        """Suggest explanation using AI and historical patterns."""
        try:
            if not self.client:
                return self._fallback_explanation(description)

            # First try hybrid prediction
            similar_transactions = self.hybrid_predictor.get_keyword_suggestions(description)
            if similar_transactions:
                best_match = max(similar_transactions, key=lambda x: x.get('confidence', 0))
                if best_match.get('confidence', 0) > 0.8:
                    return best_match.get('category', '')

            # If no good match, use OpenAI
            prompt = f"""Analyze this financial transaction and suggest a clear explanation:
            Transaction: {description}

            Provide a brief, professional explanation of the transaction purpose.
            Keep it concise (max 100 characters) and focus on the business context."""

            response = self.client.messages.create(
                model="claude-opus-4-7",
                max_tokens=100,
                system="You are a financial transaction analyst.",
                messages=[{"role": "user", "content": prompt}]
            )

            if response.content:
                return response.content[0].text.strip()

            return None

        except Exception as e:
            logger.error(f"Error suggesting explanation: {str(e)}")
            return self._fallback_explanation(description)

    def _suggest_account(self, description: str, explanation: str, amount: float) -> Optional[str]:
        """Suggest account using hybrid approach."""
        try:
            # Use existing hybrid predictor for account suggestions
            suggestions = self.hybrid_predictor.get_suggestions(
                description=description,
                amount=amount,
                historical_data=[],  # Will be populated from database
                available_accounts=[]  # Will be populated from database
            )

            if suggestions:
                best_match = max(suggestions, key=lambda x: x.get('confidence', 0))
                if best_match.get('confidence', 0) > 0.7:
                    return best_match.get('account_name')

            return None

        except Exception as e:
            logger.error(f"Error suggesting account: {str(e)}")
            return None

    def _fallback_explanation(self, description: str) -> Optional[str]:
        """Generate basic explanation when AI is unavailable."""
        try:
            # Use basic keyword matching
            keywords = {
                'salary': 'Monthly salary payment',
                'rent': 'Rental payment',
                'utilities': 'Utility bill payment',
                'invoice': 'Invoice payment',
                'refund': 'Payment refund'
            }

            description_lower = description.lower()
            for keyword, explanation in keywords.items():
                if keyword in description_lower:
                    return explanation

            return None

        except Exception as e:
            logger.error(f"Error in fallback explanation: {str(e)}")
            return None

    def enhance_historical_data(self, data: List[Dict]) -> List[Dict]:
        """Enhance historical data with AI suggestions where needed."""
        try:
            enhanced_data = []

            for entry in data:
                suggestions = self.suggest_missing_details(entry)

                if suggestions:
                    entry['ai_suggestions'] = suggestions
                    entry['has_suggestions'] = True
                else:
                    entry['has_suggestions'] = False

                enhanced_data.append(entry)

            return enhanced_data

        except Exception as e:
            logger.error(f"Error enhancing historical data: {str(e)}")
            return data