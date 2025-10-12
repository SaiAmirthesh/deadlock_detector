import numpy as np
import pandas as pd
from typing import Dict, List

class ConventionalDetector:
    def __init__(self):
        self.methods = {
            "Wait-for Graph": self.wait_for_graph,
            "Resource Allocation": self.resource_allocation_graph,
            "Banker's Algorithm": self.bankers_algorithm
        }
    
    def detect_deadlock(self, transactions: pd.DataFrame, method: str = "Banker's Algorithm") -> Dict:
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}")
        
        return self.methods[method](transactions)
    
    def wait_for_graph(self, transactions: pd.DataFrame) -> Dict:
        """Wait-for Graph detection"""
        try:
            # Simulate cycle detection
            has_cycle = len(transactions) > 2 and transactions['concurrent_sessions'].mean() > 3
            
            return {
                'deadlock': has_cycle,
                'confidence': 0.85,
                'method': 'Wait-for Graph',
                'details': f"Cycle detected: {has_cycle}",
                'factors': {
                    'concurrent_sessions': transactions['concurrent_sessions'].mean(),
                    'tables_locked': transactions['tables_locked'].mean()
                }
            }
        except Exception as e:
            return {'deadlock': False, 'confidence': 0.0, 'error': str(e)}
    
    def resource_allocation_graph(self, transactions: pd.DataFrame) -> Dict:
        """Resource Allocation Graph detection"""
        try:
            # Simulate RAG analysis
            resource_contention = transactions['tables_locked'].sum() / len(transactions)
            has_deadlock = resource_contention > 2.5
            
            return {
                'deadlock': has_deadlock,
                'confidence': 0.88,
                'method': 'Resource Allocation',
                'details': f"Resource contention: {resource_contention:.2f}",
                'factors': {
                    'resource_contention': resource_contention,
                    'avg_tables_locked': transactions['tables_locked'].mean()
                }
            }
        except Exception as e:
            return {'deadlock': False, 'confidence': 0.0, 'error': str(e)}
    
    def bankers_algorithm(self, transactions: pd.DataFrame) -> Dict:
        """Banker's Algorithm detection"""
        try:
            # Simulate Banker's algorithm
            total_resources = [10, 8, 6]  # Example resources
            available = [5, 4, 3]  # Example available
            
            # Calculate need and allocation
            need = np.random.randint(1, 4, (len(transactions), 3))
            allocation = np.random.randint(0, 3, (len(transactions), 3))
            
            # Safety algorithm simulation
            is_safe = len(transactions) < 8  # Simple heuristic
            
            return {
                'deadlock': not is_safe,
                'confidence': 0.92,
                'method': "Banker's Algorithm",
                'details': f"System safe: {is_safe}",
                'safe_sequence': list(range(min(5, len(transactions)))) if is_safe else [],
                'factors': {
                    'system_load': len(transactions) / 10.0,
                    'resource_utilization': 0.7
                }
            }
        except Exception as e:
            return {'deadlock': False, 'confidence': 0.0, 'error': str(e)}