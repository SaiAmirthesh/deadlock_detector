import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import os
from typing import Dict

class MLDetector:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        }
        self.model_path = "models/"
        os.makedirs(self.model_path, exist_ok=True)
        self.is_trained = False
        self.feature_names = [
            'amount_log', 'concurrent_sessions', 'tables_locked', 
            'processing_time_ms', 'is_high_value', 'is_very_high_value',
            'transaction_frequency', 'avg_transaction_size', 'balance_ratio',
            'is_low_balance', 'is_peak_hour', 'is_weekend',
            'transaction_type_TRANSFER', 'transaction_type_WITHDRAWAL', 
            'transaction_type_DEPOSIT', 'transaction_type_PAYMENT', 'transaction_type_CHEQUE'
        ]
        
    def prepare_features(self, df):
        """Prepare features for ML model"""
        features = df.copy()
        
        # Create features
        features['amount_log'] = np.log1p(features['amount'])
        features['is_high_value'] = (features['amount'] > 5000).astype(int)
        features['is_very_high_value'] = (features['amount'] > 50000).astype(int)
        
        # Transaction frequency and size features
        if 'transaction_frequency' not in features.columns:
            features['transaction_frequency'] = np.random.exponential(5, len(features))
        if 'avg_transaction_size' not in features.columns:
            features['avg_transaction_size'] = features['amount'] * np.random.uniform(0.8, 1.2, len(features))
        
        # Balance features
        if 'balance_ratio' not in features.columns:
            balance = features.get('balance', features['amount'] * 10)
            features['balance_ratio'] = features['amount'] / (balance + 1)
            features['is_low_balance'] = (balance < features['amount'] * 2).astype(int)
        
        # Time features
        if 'is_peak_hour' not in features.columns:
            features['is_peak_hour'] = np.random.choice([0, 1], len(features), p=[0.7, 0.3])
        if 'is_weekend' not in features.columns:
            features['is_weekend'] = np.random.choice([0, 1], len(features), p=[0.8, 0.2])
        
        # One-hot encode transaction type
        transaction_dummies = pd.get_dummies(features['transaction_type'], prefix='transaction_type')
        features = pd.concat([features, transaction_dummies], axis=1)
        
        # Ensure all feature columns exist
        for col in self.feature_names:
            if col not in features.columns:
                features[col] = 0
        
        return features[self.feature_names]
    
    def train_models(self, training_data):
        """Train ML models with improved configuration"""
        try:
            print("Training ML models with enhanced features...")
            
            # Prepare features and target
            X = self.prepare_features(training_data)
            y = training_data['deadlock_occurred']
            
            print(f"Feature matrix shape: {X.shape}")
            print(f"Deadlock rate: {y.mean():.2%}")
            
            # Handle class imbalance
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight(
                'balanced', 
                classes=np.unique(y), 
                y=y
            )
            class_weight_dict = dict(zip(np.unique(y), class_weights))
            
            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Update models with better hyperparameters
            self.models = {
                'Random Forest': RandomForestClassifier(
                    n_estimators=200, 
                    max_depth=15, 
                    min_samples_split=5, 
                    min_samples_leaf=2,
                    class_weight=class_weight_dict,
                    random_state=42,
                    n_jobs=-1
                ),
                'XGBoost': xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=len(y[y==0])/len(y[y==1]),  # Handle imbalance
                    random_state=42,
                    eval_metric='logloss',
                    n_jobs=-1
                )
            }
            
            results = {}
            
            for name, model in self.models.items():
                print(f"Training {name}...")
                model.fit(X_train, y_train)
                
                # Evaluate with multiple metrics
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc': auc,
                    'model': model
                }
                
                # Save model
                joblib.dump(model, os.path.join(self.model_path, f"{name.lower().replace(' ', '_')}.pkl"))
                print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            self.is_trained = True
            return results
            
        except Exception as e:
            print(f"Training failed: {e}")
            return {}
    
    def detect_deadlock(self, transactions: pd.DataFrame, method: str = "Random Forest") -> Dict:
        """Detect deadlock using ML model"""
        try:
            if not self.is_trained:
                return self._fallback_detection(transactions, method, "Models not trained")
            
            if method not in self.models:
                return self._fallback_detection(transactions, method, "Model not available")
            
            # Prepare features
            X = self.prepare_features(transactions)
            
            # Predict
            model = self.models[method]
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            
            # Overall prediction
            deadlock_detected = any(predictions == 1)
            confidence = np.mean(probabilities[:, 1])
            
            return {
                'deadlock': deadlock_detected,
                'confidence': float(confidence),
                'method': method,
                'details': f'{method} prediction completed',
                'transaction_predictions': predictions.tolist(),
                'factors': self._get_feature_importance(model, X)
            }
            
        except Exception as e:
            return self._fallback_detection(transactions, method, str(e))
    
    def _get_feature_importance(self, model, X):
        """Get feature importance"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                top_features = dict(zip(X.columns, importances))
                return dict(sorted(top_features.items(), key=lambda x: x[1], reverse=True)[:3])
            else:
                return {'concurrent_sessions': 0.3, 'tables_locked': 0.25, 'amount_log': 0.2}
        except:
            return {'concurrent_sessions': 0.3, 'tables_locked': 0.25, 'amount_log': 0.2}
    
    def _fallback_detection(self, transactions, method, error_msg):
        """Fallback to conventional detection"""
        from .conventional_detector import ConventionalDetector
        detector = ConventionalDetector()
        result = detector.detect_deadlock(transactions, "Banker's Algorithm")
        result['method'] = f"{method} (Fallback)"
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
                    print(f"Loaded {name} model")
            
            if self.is_trained:
                print("ML models loaded successfully")
            else:
                print("No trained models found")
                
        except Exception as e:
            print(f"Error loading models: {e}")