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
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            self.client = None
        logger.info("PredictiveFeatures initialized with thresholds - Text: 70%, Semantic: 95%")

    def find_similar_transactions(self, description: str, explanation: str = None) -> List[Dict]:
        """
        ERF: Find similar transactions based on text and semantic similarity

        Args:
            description: Current transaction description
            explanation: Current transaction explanation (optional)

        Returns:
            List of similar transactions with their similarity scores and explanations
        """
        try:
            # Get all transactions with explanations
            transactions = Transaction.query.filter(
                Transaction.explanation.isnot(None)
            ).all()
            logger.debug(f"Found {len(transactions)} transactions with explanations")

            similar_transactions = []
            logger.info(f"Finding similar transactions for description: {description}")

            # Calculate semantic embeddings if client available and explanation provided
            current_embedding = None
            
            logger.debug(f"Client available: {self.client is not None}, explanation provided: {bool(explanation)}")
            if self.client and explanation:
                try:
                    response = self.client.embeddings.create(
                        model="text-embedding-ada-002",  # Updated to use text-embedding-ada-002
                        input=explanation,
                        encoding_format="float"
                    )
                    current_embedding = response.data[0].embedding
                    logger.info("Successfully generated embedding for current explanation: ")
                except Exception as e:
                    logger.error(f"Error getting embeddings: {str(e)}")
                    current_embedding = None
        
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
                logger.debug(f"Text similarity: '{description}' vs '{transaction.description}' = {text_ratio:.2f}")

                semantic_ratio = 1.0  # Default if no semantic comparison possible

                # Calculate semantic similarity if embeddings available
                if current_embedding and transaction.explanation:
                    try:
                        response = self.client.embeddings.create(
                            model="text-embedding-ada-002",
                            input=transaction.explanation,
                            encoding_format="float"
                        )
                        tx_embedding = response.data[0].embedding
                        semantic_ratio = float(np.dot(current_embedding, tx_embedding))
                        logger.info(f"Calculated semantic similarity: {semantic_ratio}")
                    except Exception as e:
                        logger.error(f"Error calculating semantic similarity: {str(e)}")
                        semantic_ratio = 1.0

                # Only include if both thresholds are met
                if (text_ratio >= self.text_similarity_threshold and 
                    semantic_ratio >= self.semantic_similarity_threshold):
                    similar_transactions.append({
                        'id': transaction.id,
                        'description': transaction.description,
                        'explanation': transaction.explanation,
                        'text_similarity': text_ratio,
                        'semantic_similarity': semantic_ratio
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
            # Get source and target transactions
            source = Transaction.query.get(similar_transaction_id)
            target = Transaction.query.get(transaction_id)

            if not source or not target:
                logger.error("Source or target transaction not found")
                return False

            # Copy explanation
            target.explanation = source.explanation
            db.session.commit()
            logger.info(f"Successfully replicated explanation from transaction {similar_transaction_id} to {transaction_id}")
            return True

        except Exception as e:
            logger.error(f"Error replicating explanation: {str(e)}")
            db.session.rollback()
            return False

    def suggest_account(self, description: str, explanation: str) -> Dict:
        """ASF: Suggest account based on description and explanation"""
        try:
            # Get active accounts
            accounts = Account.query.filter_by(is_active=True).all()

            if not accounts:
                return {
                    'success': False,
                    'message': 'No active accounts found'
                }

            # Combine description and explanation for analysis
            combined_text = f"{description} - {explanation}"

            # Get AI suggestion if client available
            if self.client:
                try:
                    # Create prompt with account context
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
                        model="claude-opus-4-7",
                        max_tokens=256,
                        system="You are a financial account categorization expert.",
                        messages=[{"role": "user", "content": prompt}]
                    )

                    result = response.content[0].text.strip().split('|')

                    if len(result) == 3:
                        suggested_name = result[0].strip()
                        confidence = float(result[1].strip())
                        reasoning = result[2].strip()

                        # Find matching account
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

            # Fallback to basic matching
            return self._basic_account_matching(combined_text, accounts)

        except Exception as e:
            logger.error(f"Error suggesting account: {str(e)}")
            return {
                'success': False,
                'message': f'Error suggesting account: {str(e)}'
            }

    def _basic_account_matching(self, text: str, accounts: List[Account]) -> Dict:
        """Basic account matching when AI is unavailable"""
        try:
            best_match = None
            highest_similarity = 0
            reasoning = []

            for account in accounts:
                # Compare with account name and category
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