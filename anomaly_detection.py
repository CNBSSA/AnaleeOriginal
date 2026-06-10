"""
AI-Powered Anomaly Detection System
Provides automated detection of unusual patterns and anomalies in financial data
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd

from models import db, Transaction, Account, AlertConfiguration, AlertHistory
from ai_insights import FinancialInsightsGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetectionService:
    """Service for detecting anomalies in financial transactions"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.insights_generator = FinancialInsightsGenerator()
        
    def detect_anomalies(self, days_back: int = 90) -> Dict:
        """
        Main method to detect anomalies using multiple detection strategies
        
        Args:
            days_back: Number of days of historical data to analyze
            
        Returns:
            Dictionary containing detected anomalies and analysis results
        """
        try:
            # Get transaction data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            transactions = Transaction.query.filter(
                Transaction.user_id == self.user_id,
                Transaction.date.between(start_date, end_date)
            ).order_by(Transaction.date).all()
            
            if not transactions:
                return {
                    'status': 'error',
                    'message': 'Insufficient transaction data for analysis'
                }
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame([{
                'date': t.date,
                'amount': float(t.amount),
                'description': t.description,
                'category': t.account.category if t.account else 'Uncategorized',
                'transaction_id': t.id
            } for t in transactions])
            
            # Perform multiple types of anomaly detection
            statistical_anomalies = self._detect_statistical_anomalies(df)
            pattern_anomalies = self._detect_pattern_anomalies(df)
            ai_insights = self._get_ai_insights(df)
            
            # Combine and classify anomalies
            combined_anomalies = self._combine_anomaly_results(
                statistical_anomalies,
                pattern_anomalies,
                ai_insights
            )
            
            # Generate alerts for high-confidence anomalies
            self._generate_alerts(combined_anomalies)
            
            return {
                'status': 'success',
                'anomalies': combined_anomalies,
                'analysis_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                'summary': {
                    'total_transactions': len(df),
                    'anomalies_detected': len(combined_anomalies),
                    'high_risk_count': sum(1 for a in combined_anomalies if a['risk_level'] == 'high')
                }
            }
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error detecting anomalies: {str(e)}'
            }
    
    def _detect_statistical_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect anomalies using statistical methods"""
        try:
            # Prepare features for anomaly detection
            features = df[['amount']].copy()
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Use Isolation Forest for anomaly detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(scaled_features)
            
            # Get anomaly scores
            scores = iso_forest.score_samples(scaled_features)
            
            anomalies = []
            for idx, (pred, score) in enumerate(zip(predictions, scores)):
                if pred == -1:  # Anomaly detected
                    transaction = df.iloc[idx]
                    anomalies.append({
                        'transaction_id': int(transaction.transaction_id),
                        'date': transaction.date.isoformat(),
                        'amount': float(transaction.amount),
                        'description': transaction.description,
                        'anomaly_score': float(score),
                        'detection_method': 'statistical',
                        'reason': 'Unusual transaction amount'
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in statistical anomaly detection: {str(e)}")
            return []
    
    def _detect_pattern_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect anomalies based on transaction patterns"""
        try:
            anomalies = []
            
            # Group by category and calculate statistics
            category_stats = df.groupby('category')['amount'].agg(['mean', 'std']).reset_index()
            
            for _, row in df.iterrows():
                category_mean = category_stats[
                    category_stats['category'] == row['category']
                ]['mean'].values[0]
                category_std = category_stats[
                    category_stats['category'] == row['category']
                ]['std'].values[0]
                
                # Check for transactions significantly different from category average
                z_score = abs(row['amount'] - category_mean) / category_std if category_std > 0 else 0
                if z_score > 3:  # More than 3 standard deviations
                    anomalies.append({
                        'transaction_id': int(row['transaction_id']),
                        'date': row['date'].isoformat(),
                        'amount': float(row['amount']),
                        'description': row['description'],
                        'anomaly_score': float(z_score),
                        'detection_method': 'pattern',
                        'reason': f'Unusual amount for {row["category"]} category'
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in pattern anomaly detection: {str(e)}")
            return []
    
    def _get_ai_insights(self, df: pd.DataFrame) -> List[Dict]:
        """Get AI-powered insights about anomalies"""
        try:
            # Prepare data for AI analysis
            data_for_ai = [{
                'date': row['date'].isoformat(),
                'amount': row['amount'],
                'description': row['description'],
                'category': row['category']
            } for _, row in df.iterrows()]
            
            # Generate insights using AI service
            insights = self.insights_generator.generate_transaction_insights(data_for_ai)
            
            # Extract anomalies from AI insights
            anomalies = []
            for insight in insights.get('risk_factors', []):
                if 'transaction_id' in insight:
                    anomalies.append({
                        'transaction_id': insight['transaction_id'],
                        'date': insight['date'],
                        'amount': float(insight['amount']),
                        'description': insight['description'],
                        'anomaly_score': float(insight.get('risk_score', 0.5)),
                        'detection_method': 'ai',
                        'reason': insight.get('reason', 'AI-detected anomaly')
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error getting AI insights: {str(e)}")
            return []
    
    def _combine_anomaly_results(
        self,
        statistical_anomalies: List[Dict],
        pattern_anomalies: List[Dict],
        ai_anomalies: List[Dict]
    ) -> List[Dict]:
        """Combine and classify anomalies from different detection methods"""
        try:
            # Combine all anomalies
            all_anomalies = []
            seen_transactions = set()
            
            for anomaly_list in [statistical_anomalies, pattern_anomalies, ai_anomalies]:
                for anomaly in anomaly_list:
                    transaction_id = anomaly['transaction_id']
                    if transaction_id not in seen_transactions:
                        seen_transactions.add(transaction_id)
                        
                        # Calculate risk level based on detection methods and scores
                        detection_count = sum(
                            1 for lst in [statistical_anomalies, pattern_anomalies, ai_anomalies]
                            if any(a['transaction_id'] == transaction_id for a in lst)
                        )
                        
                        risk_level = 'high' if detection_count > 1 else 'medium'
                        
                        all_anomalies.append({
                            **anomaly,
                            'risk_level': risk_level,
                            'detection_count': detection_count
                        })
            
            # Sort by risk level and anomaly score
            return sorted(
                all_anomalies,
                key=lambda x: (x['risk_level'] == 'high', x['anomaly_score']),
                reverse=True
            )
            
        except Exception as e:
            logger.error(f"Error combining anomaly results: {str(e)}")
            return []
    
    def _generate_alerts(self, anomalies: List[Dict]) -> None:
        """Generate alerts for detected anomalies"""
        try:
            # Get alert configurations for the user
            alert_configs = AlertConfiguration.query.filter_by(
                user_id=self.user_id,
                is_active=True
            ).all()
            
            if not alert_configs:
                logger.info("No active alert configurations — skipping alert creation")
                return

            default_config = alert_configs[0]

            for anomaly in anomalies:
                if anomaly['risk_level'] == 'high':
                    # Create alert history entry with required FK (alert_config_id NOT NULL)
                    alert = AlertHistory(
                        alert_config_id=default_config.id,
                        user_id=self.user_id,
                        alert_message=f"High-risk anomaly detected: {anomaly['reason']} "
                                    f"(Transaction ID: {anomaly['transaction_id']})",
                        severity='high'
                    )
                    db.session.add(alert)
            
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Error generating alerts: {str(e)}")
            db.session.rollback()

