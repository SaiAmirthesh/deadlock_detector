import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Set
import logging

logger = logging.getLogger(__name__)

class ConventionalDetector:
    def __init__(self):
        self.methods = {
            "Wait-for Graph": self.wait_for_graph,
            "Resource Allocation": self.resource_allocation_graph,
            "Banker's Algorithm": self.bankers_algorithm
        }
    
    def detect_deadlock(self, processes: List[Dict], available: List[int], method: str = "Wait-for Graph") -> Tuple[bool, Dict]:
        """
        Detect deadlock using conventional methods
        """
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}")
        
        return self.methods[method](processes, available)
    
    def wait_for_graph(self, processes: List[Dict], available: List[int]) -> Tuple[bool, Dict]:
        """
        Create wait-for graph and check for cycles
        """
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add process nodes
        for process in processes:
            G.add_node(f"P{process['id']}")
        
        # Add edges based on resource needs
        for i, process in enumerate(processes):
            for j, need in enumerate(process["need"]):
                if need > 0 and available[j] < need:
                    # This process is waiting for resource j
                    # Find which process holds resource j
                    for k, other_process in enumerate(processes):
                        if i != k and other_process["allocation"][j] > 0:
                            G.add_edge(f"P{process['id']}", f"P{other_process['id']}")
                            break
        
        # Check for cycles
        try:
            cycle = nx.find_cycle(G)
            return True, {"cycle": cycle, "graph": G}
        except nx.NetworkXNoCycle:
            return False, {"graph": G}
    
    def resource_allocation_graph(self, processes: List[Dict], available: List[int]) -> Tuple[bool, Dict]:
        """
        Create resource allocation graph and check for cycles
        """
        G = nx.DiGraph()
        num_resources = len(available)
        
        # Add nodes
        for process in processes:
            G.add_node(f"P{process['id']}", type="process")
        
        for j in range(num_resources):
            G.add_node(f"R{j}", type="resource")
        
        # Add edges
        for process in processes:
            for j in range(num_resources):
                # Allocation edges (resource -> process)
                if process["allocation"][j] > 0:
                    G.add_edge(f"R{j}", f"P{process['id']}")
                
                # Request edges (process -> resource)
                if process["need"][j] > 0 and available[j] < process["need"][j]:
                    G.add_edge(f"P{process['id']}", f"R{j}")
        
        # Check for cycles
        try:
            cycle = nx.find_cycle(G)
            return True, {"cycle": cycle, "graph": G}
        except nx.NetworkXNoCycle:
            return False, {"graph": G}
    
    def bankers_algorithm(self, processes: List[Dict], available: List[int]) -> Tuple[bool, Dict]:
        """
        Implement Banker's algorithm for deadlock detection
        """
        work = available.copy()
        finish = [False] * len(processes)
        safe_sequence = []
        
        # Calculate need matrix
        need = []
        for process in processes:
            need.append([process["max"][j] - process["allocation"][j] for j in range(len(available))])
        
        # Safety algorithm
        while True:
            found = False
            for i in range(len(processes)):
                if not finish[i] and all(need[i][j] <= work[j] for j in range(len(work))):
                    # Process i can be executed
                    for j in range(len(work)):
                        work[j] += processes[i]["allocation"][j]
                    finish[i] = True
                    safe_sequence.append(f"P{processes[i]['id']}")
                    found = True
            
            if not found:
                break
        
        # Check if all processes finished
        if all(finish):
            return False, {"sequence": safe_sequence}
        else:
            return True, {"sequence": []}
    
    def detect_from_matrices(self, allocation, max_demand, available):
        """
        Detect deadlock from matrices directly (for ML training)
        """
        num_processes = allocation.shape[0]
        num_resources = allocation.shape[1]
        
        # Calculate need matrix
        need = max_demand - allocation
        
        # Banker's algorithm
        work = available.copy()
        finish = np.zeros(num_processes, dtype=bool)
        
        for _ in range(num_processes):
            found = False
            for i in range(num_processes):
                if not finish[i] and np.all(need[i] <= work):
                    work += allocation[i]
                    finish[i] = True
                    found = True
                    break
            if not found:
                return True  # Deadlock detected
        
        return False  # No deadlock