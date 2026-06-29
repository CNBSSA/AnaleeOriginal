from typing import List, Dict, Optional
import re
import logging
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
from models import db, KeywordRule

logger = logging.getLogger(__name__)

class RuleManager:
    """Manages the storage and retrieval of keyword-based rules"""
    
    def __init__(self):
        self._cached_rules = None
        self._cache_timestamp = None
        self.cache_lifetime = 300  # 5 minutes cache lifetime
        
    def add_rule(self, keyword: str, category: str, priority: int = 1,
                 is_regex: bool = False, is_active: bool = True) -> bool:
        """
        Add a new keyword rule to the database
        Returns True if successful, False otherwise
        """
        try:
            # Validate regex pattern if applicable
            if is_regex:
                try:
                    re.compile(keyword)
                except re.error as e:
                    logger.error(f"Invalid regex pattern '{keyword}': {str(e)}")
                    return False

            normalized_keyword = keyword.lower().strip()
            normalized_category = category.strip()

            # Idempotency guard: a rule is uniquely identified by
            # (keyword, category, is_regex). If one already exists, don't insert a
            # duplicate row — reactivate it if it had been deactivated, then return
            # success. This keeps repeated default-rule seeding (HybridPredictor
            # seeds defaults on every init) and any other caller from accumulating
            # duplicate KeywordRule rows.
            existing = KeywordRule.query.filter_by(
                keyword=normalized_keyword,
                category=normalized_category,
                is_regex=is_regex,
            ).first()
            if existing:
                if is_active and not existing.is_active:
                    existing.is_active = True
                    db.session.commit()
                    self._cached_rules = None
                return True

            # Create new rule
            rule = KeywordRule(
                keyword=normalized_keyword,
                category=normalized_category,
                priority=priority,
                is_regex=is_regex,
                is_active=is_active,
                created_at=datetime.utcnow()
            )
            
            db.session.add(rule)
            db.session.commit()
            
            # Invalidate cache
            self._cached_rules = None
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Database error adding rule: {str(e)}")
            db.session.rollback()
            return False
            
    def get_active_rules(self) -> List[Dict]:
        """Get all active rules with caching"""
        if (self._cached_rules is not None and 
            self._cache_timestamp is not None and
            (datetime.utcnow() - self._cache_timestamp).total_seconds() < self.cache_lifetime):
            return self._cached_rules
            
        try:
            rules = KeywordRule.query.filter_by(is_active=True).order_by(
                KeywordRule.priority.desc()
            ).all()
            
            self._cached_rules = [
                {
                    'id': rule.id,
                    'keyword': rule.keyword,
                    'category': rule.category,
                    'priority': rule.priority,
                    'is_regex': rule.is_regex,
                    'created_at': rule.created_at
                }
                for rule in rules
            ]
            self._cache_timestamp = datetime.utcnow()
            
            return self._cached_rules
            
        except SQLAlchemyError as e:
            logger.error(f"Database error fetching rules: {str(e)}")
            return []
            
    def deactivate_rule(self, rule_id: int) -> bool:
        """
        Deactivate a rule instead of deleting it
        Returns True if successful, False otherwise
        """
        try:
            rule = KeywordRule.query.get(rule_id)
            if rule:
                rule.is_active = False
                db.session.commit()
                self._cached_rules = None
                return True
            return False
            
        except SQLAlchemyError as e:
            logger.error(f"Database error deactivating rule: {str(e)}")
            db.session.rollback()
            return False
            
    def update_rule_priority(self, rule_id: int, new_priority: int) -> bool:
        """
        Update the priority of a rule
        Returns True if successful, False otherwise
        """
        try:
            rule = KeywordRule.query.get(rule_id)
            if rule:
                rule.priority = new_priority
                db.session.commit()
                self._cached_rules = None
                return True
            return False
            
        except SQLAlchemyError as e:
            logger.error(f"Database error updating rule priority: {str(e)}")
            db.session.rollback()
            return False

    def get_rule_statistics(self) -> Dict:
        """Get statistics about the rules"""
        try:
            total_rules = KeywordRule.query.count()
            active_rules = KeywordRule.query.filter_by(is_active=True).count()
            regex_rules = KeywordRule.query.filter_by(is_regex=True).count()
            
            return {
                'total_rules': total_rules,
                'active_rules': active_rules,
                'regex_rules': regex_rules,
                'cached_rules': len(self._cached_rules) if self._cached_rules else 0,
                'cache_age': (datetime.utcnow() - self._cache_timestamp).total_seconds()
                if self._cache_timestamp else None
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting rule statistics: {str(e)}")
            return {
                'total_rules': 0,
                'active_rules': 0,
                'regex_rules': 0,
                'cached_rules': 0,
                'cache_age': None
            }
