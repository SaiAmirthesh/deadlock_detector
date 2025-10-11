import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class BankersAlgorithm:
    def __init__(self):
        self.name = "Banker's Algorithm"
    
    def detect_deadlock(self, transactions: pd.DataFrame) -> Dict:
        """Traditional Banker's Algorithm for bank transactions"""
        try:
            num_processes = len(transactions)
            if num_processes == 0:
                return {'deadlock': False, 'confidence': 1.0, 'details': 'No transactions'}
            
            # Simulate resource management
            available = self._calculate_available_resources(transactions)
            max_demand = self._calculate_max_demand(transactions)
            allocation = self._calculate_allocation(transactions)
            need = max_demand - allocation
            
            # Banker's safety algorithm
            work = available.copy()
            finish = [False] * num_processes
            safe_sequence = []
            
            for _ in range(num_processes):
                found = False
                for i in range(num_processes):
                    if not finish[i] and self._can_allocate(need[i], work):
                        work += allocation[i]
                        finish[i] = True
                        safe_sequence.append(i)
                        found = True
                        break
                
                if not found:
                    break
            
            is_safe = all(finish)
            confidence = self._calculate_confidence(transactions, is_safe)
            
            return {
                'deadlock': not is_safe,
                'confidence': confidence,
                'safe_sequence': safe_sequence if is_safe else [],
                'completion_rate': sum(finish) / num_processes,
                'method': 'Bankers Algorithm',
                'factors': {
                    'resource_utilization': self._calculate_utilization(allocation, available),
                    'contention_level': transactions['concurrent_sessions'].mean() / 10.0,
                    'complexity_score': transactions['tables_locked'].mean() / 5.0
                }
            }
        except Exception as e:
            return {'deadlock': False, 'confidence': 0.0, 'error': str(e)}
    
    def _calculate_available_resources(self, transactions):
        """Calculate available system resources"""
        max_sessions = 50
        max_tables = 20
        max_connections = 100
        
        used_sessions = transactions['concurrent_sessions'].sum()
        used_tables = transactions['tables_locked'].sum()
        used_connections = len(transactions) * 2  # Approximate connections
        
        return np.array([
            max(1, max_sessions - used_sessions),
            max(1, max_tables - used_tables),
            max(1, max_connections - used_connections)
        ])
    
    def _calculate_max_demand(self, transactions):
        """Calculate maximum resource demand"""
        demands = []
        for _, tx in transactions.iterrows():
            session_demand = min(tx['concurrent_sessions'] * 1.5, 10)
            table_demand = min(tx['tables_locked'] + 1, 6)
            connection_demand = 2  # Base connection demand
            demands.append([session_demand, table_demand, connection_demand])
        
        return np.array(demands)
    
    def _calculate_allocation(self, transactions):
        """Calculate current resource allocation"""
        allocations = []
        for _, tx in transactions.iterrows():
            session_alloc = tx['concurrent_sessions']
            table_alloc = tx['tables_locked']
            connection_alloc = 1  # Base connection allocation
            allocations.append([session_alloc, table_alloc, connection_alloc])
        
        return np.array(allocations)
    
    def _can_allocate(self, need, work):
        """Check if resources can be allocated"""
        return all(need <= work)
    
    def _calculate_confidence(self, transactions, is_safe):
        """Calculate confidence score"""
        base_confidence = 0.95  # High confidence for deterministic algorithm
        
        # Adjust based on system load
        load_factor = transactions['concurrent_sessions'].mean() / 10.0
        complexity_factor = transactions['tables_locked'].mean() / 5.0
        
        confidence = base_confidence * (1 - load_factor * 0.2) * (1 - complexity_factor * 0.1)
        return max(0.7, min(confidence, 1.0))
    
    def _calculate_utilization(self, allocation, available):
        """Calculate resource utilization"""
        total_allocated = np.sum(allocation, axis=0)
        total_resources = total_allocated + available
        return np.mean(total_allocated / total_resources)