"""
Predictive Features Module for Analyze Data Menu
Implements the three key predictive features:
1. Explanation Recognition Feature (ERF)
2. Account Suggestion Feature (ASF)
3. Explanation Suggestion Feature (ESF)
"""

import logging
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
import numpy as np
from sqlalchemy import text
from models import db, Transaction, Account
from nlp_utils import get_claude_client as get_openai_client
from config import CLAUDE_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictiveFeatures:
    """Handles all predictive features for transaction analysis"""

    def __init__(self):
        self.text_similarity_threshold = 0.70  # 70% text similarity
        self.semantic_similarity_threshold = 0.95  # 95% semantic similarity
        try:
            self.client = get_openai_client()
            logger.info("AI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI client: {str(e)}")
            self.client = None
        logger.info("PredictiveFeatures initialized with thresholds - Text: 70%, Semantic: 95%")

    def find_similar_transactions(self, description: str, explanation: str = None) -> List[Dict]:
        """
        ERF: Find similar transactions based on text similarity

        Args:
            description: Current transaction description
            explanation: Current transaction explanation (optional)

        Returns:
            Dict with similar transactions and their similarity scores
        """
        try:
            # Get all transactions with explanations
            transactions = Transaction.query.filter(
                Transaction.explanation.isnot(None)
            ).all()
            logger.debug(f"Found {len(transactions)} transactions with explanations")

            similar_transactions = []
            logger.info(f"Finding similar transactions for description: {description}")

            for transaction in transactions:
                # Skip if it's the same transaction
                if transaction.description == description:
                    continue

                # Calculate text similarity
                text_ratio = SequenceMatcher(
                    None,
                    description.lower(),
                    transaction.description.lower()
                ).ratio()

                if text_ratio >= self.text_similarity_threshold:
                    similar_transactions.append({
                        'id': transaction.id,
                        'description': transaction.description,
                        'explanation': transaction.explanation,
                        'text_similarity': text_ratio,
                        'semantic_similarity': 1.0
                    })

            logger.info(f"Found {len(similar_transactions)} similar transactions")
            return {
                'success': True,
                'similar_transactions': similar_transactions
            }

        except Exception as e:
            logger.error(f"Error finding similar transactions: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'similar_transactions': []
            }

    def replicate_explanation(self, transaction_id: int, similar_transaction_id: int) -> bool:
        """
        ERF: Replicate explanation from a similar transaction to the current one

        Args:
            transaction_id: ID of the transaction to update
            similar_transaction_id: ID of the transaction to copy explanation from

        Returns:
            Boolean indicating success
        """
        try:
            source = Transaction.query.get(similar_transaction_id)
            target = Transaction.query.get(transaction_id)

            if not source or not target:
                logger.error("Source or target transaction not found")
                return False

            target.explanation = source.explanation
            db.session.commit()
            logger.info(f"Successfully replicated explanation from {similar_transaction_id} to {transaction_id}")
            return True

        except Exception as e:
            logger.error(f"Error replicating explanation: {str(e)}")
            db.session.rollback()
            return False

    def suggest_account(self, description: str, explanation: str) -> Dict:
        """ASF: Suggest account based on description and explanation"""
        try:
            accounts = Account.query.filter_by(is_active=True).all()

            if not accounts:
                return {
                    'success': False,
                    'message': 'No active accounts found'
                }

            combined_text = f"{description} - {explanation}"

            if self.client:
                try:
                    account_context = "\n".join([
                        f"- {acc.name} (Category: {acc.category})"
                        for acc in accounts
                    ])

                    prompt = f"""Analyze this financial transaction and suggest the most appropriate account:
                    Transaction: {combined_text}

                    Available accounts:
                    {account_context}

                    Respond with:
                    1. Most appropriate account name
                    2. Confidence score (0-1)
                    3. Detailed reasoning

                    Format: account|confidence|reasoning"""

                    response = self.client.messages.create(
                        model=CLAUDE_MODEL,
                        max_tokens=256,
                        system="You are a financial account categorization expert.",
                        messages=[{"role": "user", "content": prompt}]
                    )

                    result = response.content[0].text.strip().split('|')

                    if len(result) == 3:
                        suggested_name = result[0].strip()
                        confidence = float(result[1].strip())
                        reasoning = result[2].strip()

                        for account in accounts:
                            if account.name.lower() == suggested_name.lower():
                                return {
                                    'success': True,
                                    'account': account.name,
                                    'confidence': confidence,
                                    'reasoning': reasoning
                                }
                except Exception as e:
                    logger.error(f"Error getting AI suggestion: {str(e)}")

            return self._basic_account_matching(combined_text, accounts)

        except Exception as e:
            logger.error(f"Error suggesting account: {str(e)}")
            return {
                'success': False,
                'message': f'Error suggesting account: {str(e)}'
            }

    def suggest_explanation(self, description: str) -> Dict:
        """ESF: Suggest explanation based on transaction description using AI"""
        try:
            if self.client:
                try:
                    response = self.client.messages.create(
                        model=CLAUDE_MODEL,
                        max_tokens=150,
                        system="You are a financial analyst. Given a bank transaction description, "
                               "provide a concise one-sentence explanation of what this transaction "
                               "likely represents. Be specific and practical.",
                        messages=[{"role": "user", "content": f"Explain this bank transaction: {description}"}]
                    )
                    return {
                        'success': True,
                        'suggestion': response.content[0].text.strip()
                    }
                except Exception as e:
                    logger.error(f"AI explanation error: {str(e)}")

            # Fallback: find a similar explained transaction
            similar = Transaction.query.filter(
                Transaction.description.ilike(f"%{description[:20]}%"),
                Transaction.explanation.isnot(None)
            ).first()

            if similar:
                return {'success': True, 'suggestion': similar.explanation}

            return {
                'success': True,
                'suggestion': f"Transaction: {description}"
            }

        except Exception as e:
            logger.error(f"Error suggesting explanation: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _basic_account_matching(self, text: str, accounts: List[Account]) -> Dict:
        """Basic account matching when AI is unavailable"""
        try:
            best_match = None
            highest_similarity = 0
            reasoning = []

            for account in accounts:
                name_similarity = SequenceMatcher(
                    None,
                    text.lower(),
                    f"{account.name} {account.category}".lower()
                ).ratio()

                if name_similarity > highest_similarity:
                    highest_similarity = name_similarity
                    best_match = account
                    reasoning = [
                        f"Best text match with account name and category",
                        f"Similarity score: {name_similarity:.2f}",
                        f"Matched against: {account.name} ({account.category})"
                    ]

            return {
                'success': True,
                'account': best_match.name if best_match else None,
                'confidence': highest_similarity,
                'reasoning': ' | '.join(reasoning)
            }

        except Exception as e:
            logger.error(f"Error in basic account matching: {str(e)}")
            return {
                'success': False,
                'message': f'Error in basic account matching: {str(e)}'
            }
