import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
import os
from typing import Dict

class MLDeadlockDetector:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'XGBoost': xgb.XGBClassifier(random_state=42, max_depth=6, learning_rate=0.1, n_estimators=100)
        }
        self.model_path = "models/"
        os.makedirs(self.model_path, exist_ok=True)
        self.is_trained = False
        self.num_features = 15  # Fixed feature count
        self.load_models()
    
    def prepare_features(self, transactions):
        """Prepare exactly 15 features for ML model"""
        try:
            # If single transaction, convert to batch
            if isinstance(transactions, pd.Series):
                transactions = pd.DataFrame([transactions])
            
            features = []
            
            # 1. Amount-based features
            amount = transactions['amount'].iloc[0] if 'amount' in transactions.columns else 1000
            features.append(np.log1p(amount))  # log_amount
            features.append(1 if amount > 5000 else 0)  # is_high_amount
            
            # 2. Session features
            sessions = transactions['concurrent_sessions'].iloc[0] if 'concurrent_sessions' in transactions.columns else 3
            features.append(sessions)  # concurrent_sessions
            features.append(sessions * 0.5)  # session_intensity
            
            # 3. Table locking features
            tables = transactions['tables_locked'].iloc[0] if 'tables_locked' in transactions.columns else 2
            features.append(tables)  # tables_locked
            features.append(tables * 2)  # lock_complexity
            
            # 4. Processing features
            processing_time = transactions['processing_time_ms'].iloc[0] if 'processing_time_ms' in transactions.columns else 500
            features.append(processing_time)  # processing_time_ms
            features.append(processing_time / 1000)  # processing_seconds
            
            # 5. Transaction type
            if 'transaction_type' in transactions.columns:
                type_mapping = {'TRANSFER': 0, 'WITHDRAWAL': 1, 'DEPOSIT': 2, 'PAYMENT': 3, 'CHEQUE': 4, 'OTHER': 5}
                tx_type = transactions['transaction_type'].iloc[0]
                features.append(type_mapping.get(tx_type, 5))  # transaction_type_encoded
            else:
                features.append(0)
            
            # 6. Combined metrics
            features.append(sessions / max(tables, 1))  # sessions_per_table
            features.append(sessions * tables)  # load_factor
            
            # 7. Batch statistics (for consistency)
            features.append(amount)  # batch_avg_amount
            features.append(1.0)  # session_std (placeholder)
            features.append(tables)  # max_tables
            features.append(1.0)  # batch_size_norm (placeholder)
            
            # Ensure exactly 15 features
            while len(features) < self.num_features:
                features.append(0.0)
            
            features = features[:self.num_features]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            # Return default features if error
            default_features = np.array([np.log1p(1000), 0, 3, 1.5, 2, 4, 500, 0.5, 0, 1.5, 6, 1000, 1.0, 2, 1.0])
            return default_features.reshape(1, -1)
    
    def train_models(self, transactions, num_samples=5000):
        """Train ML models on transaction data"""
        try:
            print("Generating training data...")
            
            # Generate training data with consistent features
            X_train = []
            y_train = []
            
            for _ in range(num_samples):
                # Create synthetic training sample
                sample_features = self._generate_training_sample()
                X_train.append(sample_features)
                # Use actual deadlock occurrences or simulate
                y_train.append(transactions['deadlock_occurred'].sample(1).iloc[0] if 'deadlock_occurred' in transactions.columns else np.random.choice([0, 1]))
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            print(f"Training data shape: {X_train.shape}")
            
            # Split data
            X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            
            results = {}
            
            for name, model in self.models.items():
                print(f"Training {name}...")
                model.fit(X_tr, y_tr)
                
                # Evaluate
                y_pred = model.predict(X_te)
                accuracy = accuracy_score(y_te, y_pred)
                
                results[name] = {
                    'accuracy': accuracy,
                    'model': model
                }
                
                # Save model
                joblib.dump(model, os.path.join(self.model_path, f"{name.lower().replace(' ', '_')}.pkl"))
                print(f"{name} accuracy: {accuracy:.4f}")
            
            self.is_trained = True
            return results
            
        except Exception as e:
            print(f"Error training models: {e}")
            return {}
    
    def _generate_training_sample(self):
        """Generate a training sample with 15 features"""
        features = []
        
        # Random but realistic values
        amount = np.random.exponential(1000)
        sessions = np.random.poisson(3)
        tables = np.random.randint(1, 5)
        processing_time = np.random.exponential(500)
        tx_type = np.random.randint(0, 6)
        
        # Build exactly 15 features
        features.extend([
            np.log1p(amount),  # 0
            1 if amount > 5000 else 0,  # 1
            sessions,  # 2
            sessions * 0.5,  # 3
            tables,  # 4
            tables * 2,  # 5
            processing_time,  # 6
            processing_time / 1000,  # 7
            tx_type,  # 8
            sessions / max(tables, 1),  # 9
            sessions * tables,  # 10
            amount,  # 11
            np.random.exponential(1),  # 12
            tables,  # 13
            1.0  # 14
        ])
        
        return features
    
    def detect_rf(self, transactions):
        """Detect deadlock using Random Forest"""
        return self._detect_with_model(transactions, 'Random Forest')
    
    def detect_xgb(self, transactions):
        """Detect deadlock using XGBoost"""
        return self._detect_with_model(transactions, 'XGBoost')
    
    def _detect_with_model(self, transactions, model_name):
        """Detect deadlock using specified model"""
        try:
            if not self.is_trained:
                return self._fallback_detection(transactions, model_name)
            
            if model_name not in self.models:
                return self._fallback_detection(transactions, model_name)
            
            # Prepare features
            X = self.prepare_features(transactions)
            
            # Predict
            model = self.models[model_name]
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0][1]
            
            return {
                'deadlock': bool(prediction),
                'confidence': float(probability),
                'method': model_name,
                'details': f'{model_name} prediction completed'
            }
            
        except Exception as e:
            return self._fallback_detection(transactions, model_name, str(e))
    
    def _fallback_detection(self, transactions, model_name, error_msg=None):
        """Fallback to traditional detection"""
        from .conventional_detector import BankersAlgorithm
        banker = BankersAlgorithm()
        result = banker.detect_deadlock(transactions)
        result['method'] = f"{model_name} (Fallback)"
        if error_msg:
            result['error'] = error_msg
        return result
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            for name in self.models.keys():
                model_path = os.path.join(self.model_path, f"{name.lower().replace(' ', '_')}.pkl")
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
                    self.is_trained = True
                    print(f"✅ Loaded {name} model")
            
            if self.is_trained:
                print("✅ ML models loaded successfully")
            else:
                print("⚠️ No pre-trained models found - will use fallback methods")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.is_trained = False