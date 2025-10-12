import pandas as pd
import numpy as np
from typing import Dict

class HybridDetector:
    def __init__(self):
        self.name = "Enhanced Hybrid Detector"
        self.ml_detector = None
        self.conventional_detector = None
        
    def initialize_detectors(self):
        """Lazy initialization to avoid circular imports"""
        if self.ml_detector is None:
            from .ml_detector import MLDetector
            self.ml_detector = MLDetector()
            # Load pre-trained models for the new instance
            self.ml_detector.load_models()
        
        if self.conventional_detector is None:
            from .conventional_detector import ConventionalDetector
            self.conventional_detector = ConventionalDetector()
    
    def detect_deadlock(self, transactions: pd.DataFrame) -> Dict:
        """Enhanced hybrid detection combining ML and conventional methods"""
        try:
            self.initialize_detectors()
            
            # Analyze data characteristics
            data_complexity = self._calculate_complexity(transactions)
            data_size = len(transactions)
            
            # Enhanced decision logic with confidence scoring
            ml_confidence = self._calculate_ml_confidence(transactions)
            traditional_confidence = self._calculate_traditional_confidence(transactions)
            
            # Use the method with higher confidence, but with safety checks
            use_ml = (self.ml_detector.is_trained and 
                     data_size > 5 and 
                     data_complexity < 0.9 and
                     ml_confidence > traditional_confidence)
            
            if use_ml:
                # Use ML with ensemble approach
                rf_result = self.ml_detector.detect_deadlock(transactions, "Random Forest")
                xgb_result = self.ml_detector.detect_deadlock(transactions, "XGBoost")
                
                # Ensemble prediction
                ensemble_deadlock = rf_result['deadlock'] or xgb_result['deadlock']
                ensemble_confidence = (rf_result['confidence'] + xgb_result['confidence']) / 2
                
                result = {
                    'deadlock': ensemble_deadlock,
                    'confidence': ensemble_confidence,
                    'method': 'Ensemble ML',
                    'method_used': 'Random Forest + XGBoost (Ensemble)',
                    'decision_reason': f'ML ensemble with high confidence (ML: {ml_confidence:.2f} vs Traditional: {traditional_confidence:.2f})',
                    'rf_details': rf_result,
                    'xgb_details': xgb_result,
                    'factors': self._combine_feature_importance(rf_result, xgb_result)
                }
            else:
                # Use traditional method as fallback
                result = self.conventional_detector.detect_deadlock(transactions, "Banker's Algorithm")
                result['method_used'] = 'Banker\'s Algorithm (Traditional)'
                
                if not self.ml_detector.is_trained:
                    reason = 'ML models not trained - using traditional method'
                elif data_size <= 5:
                    reason = f'Small dataset ({data_size} transactions) - traditional method more reliable'
                elif data_complexity >= 0.9:
                    reason = f'Very complex dataset (complexity: {data_complexity:.2f}) - traditional method more reliable'
                else:
                    reason = f'Traditional method has higher confidence (Traditional: {traditional_confidence:.2f} vs ML: {ml_confidence:.2f})'
                
                result['decision_reason'] = reason
            
            # Add hybrid analysis info
            result['hybrid_analysis'] = {
                'data_size': data_size,
                'complexity_score': data_complexity,
                'ml_trained': self.ml_detector.is_trained,
                'ml_confidence': ml_confidence,
                'traditional_confidence': traditional_confidence,
                'final_method': result['method_used']
            }
            
            return result
            
        except Exception as e:
            # Ultimate fallback
            return {
                'deadlock': False,
                'confidence': 0.5,
                'error': str(e),
                'method_used': 'Fallback',
                'decision_reason': 'Error in hybrid detection - using safe fallback'
            }
    
    def _calculate_complexity(self, transactions):
        """Calculate complexity score of the transaction batch"""
        try:
            factors = []
            
            # Concurrent sessions complexity
            if 'concurrent_sessions' in transactions.columns:
                session_complexity = min(transactions['concurrent_sessions'].mean() / 10.0, 1.0)
                factors.append(session_complexity)
            
            # Table locking complexity
            if 'tables_locked' in transactions.columns:
                table_complexity = min(transactions['tables_locked'].mean() / 5.0, 1.0)
                factors.append(table_complexity)
            
            # Amount complexity
            if 'amount' in transactions.columns:
                amount_complexity = min(transactions['amount'].mean() / 10000.0, 1.0)
                factors.append(amount_complexity)
            
            # Transaction type complexity
            if 'transaction_type' in transactions.columns:
                type_complexity = self._calculate_type_complexity(transactions)
                factors.append(type_complexity)
            
            return np.mean(factors) if factors else 0.5
            
        except:
            return 0.5
    
    def _calculate_type_complexity(self, transactions):
        """Calculate complexity based on transaction types"""
        try:
            type_counts = transactions['transaction_type'].value_counts()
            complex_types = ['TRANSFER', 'CHEQUE', 'PAYMENT']
            
            complex_count = sum(type_counts.get(t, 0) for t in complex_types)
            complexity_ratio = complex_count / len(transactions)
            
            return min(complexity_ratio * 2, 1.0)  # Scale to 0-1
        except:
            return 0.5
    
    def _calculate_ml_confidence(self, transactions):
        """Calculate confidence score for ML approach"""
        try:
            if not self.ml_detector.is_trained:
                return 0.0
            
            # Factors that favor ML approach
            confidence = 0.5
            
            # Data size factor (more data = better for ML)
            data_size = len(transactions)
            if data_size > 100:
                confidence += 0.3
            elif data_size > 50:
                confidence += 0.2
            elif data_size > 20:
                confidence += 0.1
            
            # Feature richness factor
            feature_richness = self._calculate_feature_richness(transactions)
            confidence += feature_richness * 0.3
            
            # Data quality factor
            data_quality = self._calculate_data_quality(transactions)
            confidence += data_quality * 0.2
            
            return min(confidence, 1.0)
        except:
            return 0.0
    
    def _calculate_traditional_confidence(self, transactions):
        """Calculate confidence score for traditional approach"""
        try:
            confidence = 0.6  # Base confidence for traditional methods
            
            # Traditional methods work well with smaller datasets
            data_size = len(transactions)
            if data_size <= 10:
                confidence += 0.3
            elif data_size <= 50:
                confidence += 0.1
            
            # High complexity favors traditional methods
            complexity = self._calculate_complexity(transactions)
            if complexity > 0.7:
                confidence += 0.2
            elif complexity > 0.5:
                confidence += 0.1
            
            return min(confidence, 1.0)
        except:
            return 0.6
    
    def _calculate_feature_richness(self, transactions):
        """Calculate how rich the feature set is"""
        try:
            rich_features = ['amount', 'transaction_type', 'concurrent_sessions', 
                           'tables_locked', 'processing_time_ms']
            
            available_features = sum(1 for col in rich_features if col in transactions.columns)
            return available_features / len(rich_features)
        except:
            return 0.5
    
    def _calculate_data_quality(self, transactions):
        """Calculate data quality score"""
        try:
            # Check for missing values
            missing_ratio = transactions.isnull().sum().sum() / (len(transactions) * len(transactions.columns))
            quality_score = 1 - missing_ratio
            
            # Check for data consistency
            if 'amount' in transactions.columns:
                numeric_amounts = pd.to_numeric(transactions['amount'], errors='coerce')
                valid_amounts = numeric_amounts.notna().sum() / len(transactions)
                quality_score = (quality_score + valid_amounts) / 2
            
            return quality_score
        except:
            return 0.5
    
    def _combine_feature_importance(self, rf_result, xgb_result):
        """Combine feature importance from both models"""
        try:
            combined = {}
            
            rf_factors = rf_result.get('factors', {})
            xgb_factors = xgb_result.get('factors', {})
            
            # Combine factors with equal weighting
            all_factors = set(rf_factors.keys()) | set(xgb_factors.keys())
            for factor in all_factors:
                rf_weight = rf_factors.get(factor, 0)
                xgb_weight = xgb_factors.get(factor, 0)
                combined[factor] = (rf_weight + xgb_weight) / 2
            
            # Return top 5 factors
            return dict(sorted(combined.items(), key=lambda x: x[1], reverse=True)[:5])
        except:
            return {'concurrent_sessions': 0.3, 'tables_locked': 0.25, 'amount_log': 0.2}
