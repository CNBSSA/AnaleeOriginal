from typing import List, Dict, Optional
import re
from collections import defaultdict
import logging

from .rule_manager import RuleManager

logger = logging.getLogger(__name__)

class KeywordMatcher:
    def __init__(self):
        self.rule_manager = RuleManager()
        self.category_keywords = defaultdict(set)
        self.custom_rules = []
        self._load_rules()
    def _load_rules(self):
        """Load rules from database"""
        try:
            active_rules = self.rule_manager.get_active_rules()
            for rule in active_rules:
                if rule['is_regex']:
                    self.add_custom_rule(rule['keyword'], rule['category'], rule['priority'])
                else:
                    self.category_keywords[rule['category']].add(rule['keyword'])
            logger.info(f"Loaded {len(active_rules)} rules from database")
        except Exception as e:
            logger.error(f"Error loading rules: {str(e)}")
        
    def add_keyword_rule(self, keyword: str, account_category: str, priority: int = 1):
        """Add a keyword-based rule for account categorization and persist it"""
        keyword = keyword.lower().strip()
        if self.rule_manager.add_rule(keyword, account_category, priority=priority):
            self.category_keywords[account_category].add(keyword)
            logger.info(f"Added keyword rule: {keyword} -> {account_category}")
            return True
        return False
        
    def add_custom_rule(self, pattern: str, account_category: str, priority: int = 1):
        """Add a custom regex pattern rule"""
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            self.custom_rules.append({
                'pattern': compiled_pattern,
                'category': account_category,
                'priority': priority
            })
            self.custom_rules.sort(key=lambda x: x['priority'], reverse=True)
        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern}': {str(e)}")
            
    def find_matching_categories(self, description: str) -> List[Dict]:
        """Find matching categories based on keywords and rules"""
        description = description.lower()
        matches = []
        
        # Check custom rules first
        for rule in self.custom_rules:
            if rule['pattern'].search(description):
                matches.append({
                    'category': rule['category'],
                    'confidence': 0.9,
                    'match_type': 'custom_rule',
                    'rule_priority': rule['priority']
                })
                
        # Check keyword matches
        for category, keywords in self.category_keywords.items():
            matched_keywords = [k for k in keywords if k in description]
            if matched_keywords:
                confidence = min(len(matched_keywords) * 0.3, 0.8)
                matches.append({
                    'category': category,
                    'confidence': confidence,
                    'match_type': 'keyword',
                    'matched_keywords': matched_keywords
                })
                
        return sorted(matches, key=lambda x: x.get('confidence', 0), reverse=True)
        
    def suggest_categories(self, description: str, 
                         min_confidence: float = 0.3) -> List[Dict]:
        """Get category suggestions for a transaction description"""
        matches = self.find_matching_categories(description)
        return [m for m in matches if m['confidence'] >= min_confidence]
