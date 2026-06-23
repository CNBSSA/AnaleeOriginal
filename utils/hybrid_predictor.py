from typing import List, Dict, Optional, Tuple
import logging
from .pattern_matching import PatternMatcher
from .keyword_matcher import KeywordMatcher
from ai_utils import predict_account, calculate_text_similarity
import time

logger = logging.getLogger(__name__)

class HybridPredictor:
    def __init__(self):
        self.pattern_matcher = PatternMatcher()
        self.keyword_matcher = KeywordMatcher()
        self.confidence_threshold = 0.85
        self.use_ai_threshold = 0.7
        self._initialize_keyword_rules()
        
    def _initialize_keyword_rules(self):
        """Initialize basic keyword rules for common categories"""
        # These are just examples - actual rules should be loaded from configuration
        common_rules = {
            'Utilities': ['electricity', 'water', 'gas', 'utility'],
            'Office Supplies': ['supplies', 'paper', 'toner', 'printer'],
            'Travel': ['flight', 'hotel', 'taxi', 'uber'],
            'Maintenance': ['repair', 'maintenance', 'cleaning'],
        }
        
        for category, keywords in common_rules.items():
            for keyword in keywords:
                self.keyword_matcher.add_keyword_rule(keyword, category)
                
    def get_keyword_suggestions(self, description: str) -> List[Dict]:
        """Get suggestions based on keyword matching"""
        try:
            return self.keyword_matcher.suggest_categories(description)
        except Exception as e:
            logger.error(f"Error in keyword matching: {str(e)}")
            return []

    def get_suggestions(self,
                            description: str,
                            amount: float,
                            historical_data: List[Dict],
                            available_accounts: List[Dict]) -> List[Dict]:
        """
        Get suggestions using hybrid approach:
        1. Try pattern matching first
        2. If confidence is low, use AI predictions
        3. Combine results with confidence scores
        """
        try:
            # Start with pattern matching
            pattern_suggestions = self.pattern_matcher.suggest_from_patterns(
                description, amount, historical_data
            )
            
            # Get keyword-based suggestions
            keyword_suggestions = self.get_keyword_suggestions(description)
            
            combined = []
            
            # Add pattern-based suggestions
            for suggestion in pattern_suggestions:
                suggestion['source'] = 'pattern'
                combined.append(suggestion)
                
            # Add keyword-based suggestions
            for suggestion in keyword_suggestions:
                combined.append({
                    'confidence': suggestion['confidence'],
                    'category': suggestion['category'],
                    'match_type': suggestion['match_type'],
                    'source': 'keyword'
                })
            
            # Enhanced decision making for AI routing
            pattern_confidence = max((s.get('confidence', 0) for s in combined), default=0)
            pattern_reliability = max(
                (s.get('pattern_confidence', {}).get('reliability_score', 0) 
                 for s in combined), default=0
            )
            
            # Smart routing logic
            should_use_ai = (
                pattern_confidence < self.use_ai_threshold or
                (pattern_reliability < 0.7 and pattern_confidence < 0.9)
            )
            
            if should_use_ai:
                try:
                    ai_suggestions = predict_account(description, "", available_accounts)
                    for ai_suggestion in ai_suggestions:
                        confidence_boost = 0.1 if pattern_confidence > 0.5 else 0
                        combined.append({
                            'confidence': ai_suggestion['confidence'] + confidence_boost,
                            'account_name': ai_suggestion['account_name'],
                            'account': ai_suggestion['account'],
                            'reasoning': ai_suggestion['reasoning'],
                            'source': 'ai',
                            'hybrid_score': {
                                'pattern_confidence': pattern_confidence,
                                'ai_confidence': ai_suggestion['confidence'],
                                'reliability': pattern_reliability
                            }
                        })
                except Exception as ai_error:
                    logger.error(f"Error getting AI suggestions: {str(ai_error)}")
                    logger.debug(f"Pattern confidence: {pattern_confidence}, "
                               f"Reliability: {pattern_reliability}")
            
            # Sort by confidence and return top suggestions
            combined.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            return combined[:3]
            
        except Exception as e:
            logger.error(f"Error in hybrid prediction: {str(e)}")
            return []