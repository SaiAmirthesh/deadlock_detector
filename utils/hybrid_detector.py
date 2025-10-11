import pandas as pd
import numpy as np
from typing import Dict

class HybridDetector:
    def __init__(self):
        self.name = "Hybrid Detector"
        self.complexity_threshold = 0.6
        self.data_size_threshold = 100
    
    def detect(self, transactions):
        """Hybrid detection that chooses between ML and traditional methods"""
        try:
            # Analyze data characteristics
            complexity_score = self._calculate_complexity(transactions)
            data_size = len(transactions)
            
            # Decision logic
            if data_size > self.data_size_threshold and complexity_score < self.complexity_threshold:
                # Use ML for large, less complex datasets
                from .ml_detector import MLDeadlockDetector
                ml_detector = MLDeadlockDetector()
                result = ml_detector.detect_rf(transactions)
                result['method_used'] = 'Random Forest (ML)'
                result['decision_reason'] = f'Large dataset ({data_size} transactions) with low complexity'
            else:
                # Use traditional method for small or complex datasets
                from .conventional_detector import BankersAlgorithm
                banker = BankersAlgorithm()
                result = banker.detect_deadlock(transactions)
                result['method_used'] = 'Bankers Algorithm (Traditional)'
                result['decision_reason'] = f'Optimal for current dataset characteristics'
            
            # Add hybrid analysis info
            result['hybrid_analysis'] = {
                'data_size': data_size,
                'complexity_score': complexity_score,
                'data_size_threshold': self.data_size_threshold,
                'complexity_threshold': self.complexity_threshold
            }
            
            return result
            
        except Exception as e:
            return {
                'deadlock': False,
                'confidence': 0.5,
                'error': str(e),
                'method_used': 'Fallback',
                'decision_reason': 'Error in hybrid detection'
            }
    
    def _calculate_complexity(self, transactions):
        """Calculate complexity score of the transaction batch"""
        try:
            factors = []
            
            # Concurrent sessions complexity
            session_complexity = min(transactions['concurrent_sessions'].mean() / 10.0, 1.0)
            factors.append(session_complexity)
            
            # Table locking complexity
            table_complexity = min(transactions['tables_locked'].mean() / 5.0, 1.0)
            factors.append(table_complexity)
            
            # Amount complexity (larger amounts often mean more validations)
            amount_complexity = min(transactions['amount'].mean() / 10000.0, 1.0)
            factors.append(amount_complexity)
            
            # Processing time complexity
            time_complexity = min(transactions['processing_time_ms'].mean() / 1000.0, 1.0)
            factors.append(time_complexity * 0.5)
            
            return np.mean(factors)
            
        except:
            return 0.5