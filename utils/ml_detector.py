import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import os
import logging
import networkx as nx

logger = logging.getLogger(__name__)

class MLDetector:
    def __init__(self):
        self.models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        }
        self.model_path = "models/"
        self.is_trained = False
        self.num_features = 15

        # Try to load feature count
        try:
            self.num_features = joblib.load(os.path.join(self.model_path, "num_features.pkl"))
        except:
            pass

        self.load_models()
        
    def generate_training_data(self, num_samples=10000, max_processes=10, max_resources=5):
        """Generate synthetic training data for deadlock detection"""
        features = []
        labels = []
        
        for _ in range(num_samples):
            # Random system configuration
            num_processes = np.random.randint(2, max_processes + 1)
            num_resources = np.random.randint(2, max_resources + 1)
            
            # Generate random resource allocations
            total_resources = np.random.randint(1, 10, size=num_resources)
            allocation = np.zeros((num_processes, num_resources))
            max_demand = np.zeros((num_processes, num_resources))
            
            for i in range(num_processes):
                for j in range(num_resources):
                    max_demand[i, j] = np.random.randint(0, total_resources[j] + 1)
                    allocation[i, j] = np.random.randint(0, min(max_demand[i, j] + 1, total_resources[j] + 1))
            
            # Calculate available resources
            allocated = np.sum(allocation, axis=0)
            available = total_resources - allocated
            
            # Ensure system is in valid state (no negative available)
            if np.any(available < 0):
                continue
                
            # Calculate need matrix
            need = max_demand - allocation
            
            # Extract features
            system_features = self._extract_features(allocation, max_demand, need, available, total_resources)
            
            # Use conventional method as ground truth
            is_deadlock = self._conventional_detection(allocation, max_demand, available)
            
            features.append(system_features)
            labels.append(1 if is_deadlock else 0)
        
        return np.array(features), np.array(labels)
    
    def _extract_features(self, allocation, max_demand, need, available, total_resources):
        """Extract meaningful features from system state with fixed size"""
        features = []
        
        # Basic statistics (5 features)
        total_allocated = np.sum(allocation)
        total_capacity = np.sum(total_resources)
        features.append(total_allocated / total_capacity if total_capacity > 0 else 0)  # Resource utilization
        
        # Process waiting statistics (2 features)
        processes_waiting = np.sum(need > 0, axis=1)
        features.append(np.mean(processes_waiting))  # Avg processes waiting
        features.append(np.max(processes_waiting))   # Max processes waiting
        
        # Resource availability (2 features)
        available_ratio = available / total_resources
        features.append(np.mean(available_ratio))    # Avg available resources
        features.append(np.min(available_ratio))     # Min available resources
        
        # Wait-for graph characteristics (3 features)
        wait_for_graph = self._build_wait_for_graph(allocation, need)
        features.append(wait_for_graph.number_of_nodes())
        features.append(wait_for_graph.number_of_edges())
        
        # Check for cycles (potential deadlocks) (1 feature)
        try:
            cycle = nx.find_cycle(wait_for_graph)
            features.append(len(cycle) if cycle else 0)
        except nx.NetworkXNoCycle:
            features.append(0)
        
        # Resource contention metrics (up to 5 features, pad if needed)
        num_resources = allocation.shape[1]
        for j in range(min(num_resources, 5)):  # Limit to first 5 resources
            features.append(np.sum(need[:, j] > 0))  # Processes waiting for resource j
            features.append(available[j] / total_resources[j] if total_resources[j] > 0 else 0)
        
        # Pad with zeros if we have fewer than expected features
        while len(features) < self.num_features:
            features.append(0)
        
        # Truncate if we have more than expected features
        features = features[:self.num_features]
        
        return np.array(features)
    
    def _build_wait_for_graph(self, allocation, need):
        """Build wait-for graph from allocation and need matrices"""
        G = nx.DiGraph()
        num_processes = allocation.shape[0]
        num_resources = allocation.shape[1]
        
        # Add process nodes
        for i in range(num_processes):
            G.add_node(f"P{i}", type="process")
        
        # Add resource nodes
        for j in range(num_resources):
            G.add_node(f"R{j}", type="resource")
        
        # Add edges based on allocation and need
        for i in range(num_processes):
            for j in range(num_resources):
                if allocation[i, j] > 0:
                    # Process holds resource
                    G.add_edge(f"R{j}", f"P{i}")
                
                if need[i, j] > 0:
                    # Process needs resource
                    G.add_edge(f"P{i}", f"R{j}")
        
        return G
    
    def _conventional_detection(self, allocation, max_demand, available):
        """Simple conventional deadlock detection"""
        # Banker's algorithm simplified check
        num_processes = allocation.shape[0]
        num_resources = allocation.shape[1]
        
        # Calculate need matrix
        need = max_demand - allocation
        
        # Safety algorithm
        work = available.copy()
        finish = [False] * num_processes
        
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
    
    def train_models(self, num_samples=10000):
        """Train ML models on generated data"""
        logger.info("Generating training data...")
        X, y = self.generate_training_data(num_samples)
        
        logger.info(f"Training data shape: {X.shape}, Labels: {y.sum()} deadlocks")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                "model": model,
                "accuracy": accuracy,
                "report": classification_report(y_test, y_pred)
            }
            
            # Save model
            joblib.dump(model, os.path.join(self.model_path, f"{name.lower().replace(' ', '_')}.pkl"))
            logger.info(f"{name} accuracy: {accuracy:.4f}")
        
        self.is_trained = True
        return results
    
    def detect_deadlock(self, system_state, method="Random Forest"):
        """Detect deadlock using ML model"""
        if not self.is_trained:
            # For demo purposes, use a small sample size
            try:
                self.train_models(1000)
            except Exception as e:
                logger.error(f"Error training models: {e}")
                # Fall back to conventional detection
                from .conventional_detector import ConventionalDetector
                conventional_detector = ConventionalDetector()
                deadlock, details = conventional_detector.detect_deadlock(
                    system_state["processes"], system_state["available"], "Wait-for Graph"
                )
                return deadlock, {"method": "Fallback Conventional", "error": str(e)}
        
        if method not in self.models:
            raise ValueError(f"Unknown ML method: {method}")
        
        # Extract features from system state
        allocation = np.array([p["allocation"] for p in system_state["processes"]])
        max_demand = np.array([p["max"] for p in system_state["processes"]])
        need = np.array([p["need"] for p in system_state["processes"]])
        available = np.array(system_state["available"])
        total_resources = np.array(system_state["total_resources"])
        
        features = self._extract_features(allocation, max_demand, need, available, total_resources)
        
        # Predict
        model = self.models[method]
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0]
        
        return bool(prediction), {
            "confidence": probability[1] if prediction else probability[0],
            "features": features.tolist()
        }
    
    def load_models(self):
        """Load pre-trained models"""
        for name in self.models.keys():
            model_path = os.path.join(self.model_path, f"{name.lower().replace(' ', '_')}.pkl")
            if os.path.exists(model_path):
                try:
                    self.models[name] = joblib.load(model_path)
                    self.is_trained = True
                    logger.info(f"Loaded pre-trained {name} model")
                except Exception as e:
                    logger.error(f"Error loading model {name}: {e}")
        
        if self.is_trained:
            logger.info("ML models loaded successfully")
        else:
            logger.warning("No pre-trained models found")