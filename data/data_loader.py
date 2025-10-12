import pandas as pd
import numpy as np
import os
from datetime import datetime

class DataLoader:
    def __init__(self):
        self.data_path = "data/"
    
    def load_kaggle_data(self, file_path):
        """Load and process Kaggle dataset for training"""
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded Kaggle data: {df.shape}")
            return self.process_training_data(df)
        except Exception as e:
            print(f"Error loading Kaggle data: {e}")
            return self.generate_synthetic_data()
    
    def process_training_data(self, df):
        """Process Kaggle data for ML training"""
        processed_df = df.copy()
        
        # Feature engineering for deadlock prediction
        processed_df = self.create_features(processed_df)
        
        # Ensure we have the target variable
        if 'deadlock_occurred' not in processed_df.columns:
            processed_df['deadlock_occurred'] = self.simulate_deadlocks(processed_df)
        
        return processed_df
    
    def create_features(self, df):
        """Create advanced features for ML model"""
        features_df = df.copy()
        
        # Extract amount from existing columns
        if 'amount' not in features_df.columns:
            if 'WITHDRAWAL AMT' in features_df.columns:
                features_df['amount'] = features_df['WITHDRAWAL AMT'].fillna(0)
            elif 'DEPOSIT AMT' in features_df.columns:
                features_df['amount'] = features_df['DEPOSIT AMT'].fillna(0)
            else:
                features_df['amount'] = np.random.exponential(1000, len(features_df))
        
        # Clean amount data
        features_df['amount'] = pd.to_numeric(
            features_df['amount'].astype(str).str.replace(',', '').str.replace(' ', ''), 
            errors='coerce'
        ).fillna(0)
        
        # Extract transaction type from details
        if 'transaction_type' not in features_df.columns:
            if 'TRANSACTION DETAILS' in features_df.columns:
                features_df['transaction_type'] = features_df['TRANSACTION DETAILS'].apply(
                    self.classify_transaction_type
                )
            else:
                features_df['transaction_type'] = np.random.choice(
                    ['TRANSFER', 'WITHDRAWAL', 'DEPOSIT', 'PAYMENT'], 
                    len(features_df)
                )
        
        # Advanced feature engineering
        features_df['amount_log'] = np.log1p(features_df['amount'])
        features_df['is_high_value'] = (features_df['amount'] > 5000).astype(int)
        features_df['is_very_high_value'] = (features_df['amount'] > 50000).astype(int)
        
        # Transaction frequency features (simulate based on amount patterns)
        features_df['transaction_frequency'] = np.random.exponential(5, len(features_df))
        features_df['avg_transaction_size'] = features_df['amount'] * np.random.uniform(0.8, 1.2, len(features_df))
        
        # Resource contention features (more realistic)
        features_df['concurrent_sessions'] = np.random.poisson(4, len(features_df)) + 1
        features_df['tables_locked'] = np.random.poisson(2, len(features_df)) + 1
        features_df['processing_time_ms'] = np.random.gamma(2, 200, len(features_df))
        
        # Time-based features
        if 'DATE' in features_df.columns:
            features_df['DATE'] = pd.to_datetime(features_df['DATE'], errors='coerce')
            features_df['hour'] = features_df['DATE'].dt.hour.fillna(12)
            features_df['is_peak_hour'] = features_df['hour'].isin([9, 10, 11, 14, 15, 16]).astype(int)
            features_df['is_weekend'] = features_df['DATE'].dt.weekday.isin([5, 6]).astype(int)
        else:
            features_df['hour'] = np.random.randint(0, 24, len(features_df))
            features_df['is_peak_hour'] = np.random.choice([0, 1], len(features_df), p=[0.7, 0.3])
            features_df['is_weekend'] = np.random.choice([0, 1], len(features_df), p=[0.8, 0.2])
        
        # Account balance features
        if 'BALANCE AMT' in features_df.columns:
            features_df['balance'] = pd.to_numeric(
                features_df['BALANCE AMT'].astype(str).str.replace(',', '').str.replace(' ', ''), 
                errors='coerce'
            ).fillna(features_df['amount'].median() * 10)
        else:
            features_df['balance'] = features_df['amount'] * np.random.uniform(5, 20, len(features_df))
        
        features_df['balance_ratio'] = features_df['amount'] / (features_df['balance'] + 1)
        features_df['is_low_balance'] = (features_df['balance'] < features_df['amount'] * 2).astype(int)
        
        return features_df
    
    def simulate_deadlocks(self, df):
        """Simulate deadlock occurrences based on realistic transaction patterns"""
        deadlock_probs = []
        
        for _, row in df.iterrows():
            base_prob = 0.02  # Lower base probability for more realistic simulation
            
            # High-value transaction risk
            amount = row.get('amount', 0)
            if amount > 50000:
                base_prob += 0.25
            elif amount > 10000:
                base_prob += 0.15
            elif amount > 5000:
                base_prob += 0.08
            
            # Resource contention risk
            concurrent_sessions = row.get('concurrent_sessions', 0)
            if concurrent_sessions > 8:
                base_prob += 0.20
            elif concurrent_sessions > 5:
                base_prob += 0.12
            elif concurrent_sessions > 3:
                base_prob += 0.06
            
            # Table locking risk
            tables_locked = row.get('tables_locked', 0)
            if tables_locked > 4:
                base_prob += 0.18
            elif tables_locked > 2:
                base_prob += 0.10
            
            # Transaction type risk
            transaction_type = row.get('transaction_type', '')
            if transaction_type == 'TRANSFER':
                base_prob += 0.15
            elif transaction_type == 'PAYMENT':
                base_prob += 0.10
            elif transaction_type == 'CHEQUE':
                base_prob += 0.08
            
            # Time-based risk factors
            if row.get('is_peak_hour', 0):
                base_prob += 0.08
            
            if row.get('is_weekend', 0):
                base_prob += 0.05
            
            # Balance-related risk
            if row.get('is_low_balance', 0):
                base_prob += 0.12
            
            balance_ratio = row.get('balance_ratio', 0)
            if balance_ratio > 0.8:
                base_prob += 0.15
            elif balance_ratio > 0.5:
                base_prob += 0.08
            
            # Processing time risk (longer processing = higher deadlock risk)
            processing_time = row.get('processing_time_ms', 0)
            if processing_time > 1000:
                base_prob += 0.12
            elif processing_time > 500:
                base_prob += 0.06
            
            # Cap the probability
            base_prob = min(base_prob, 0.8)
            
            deadlock_probs.append(1 if np.random.random() < base_prob else 0)
        
        return deadlock_probs
    
    def generate_synthetic_data(self):
        """Generate synthetic data if Kaggle data is not available"""
        print("Generating synthetic training data...")
        
        n_samples = 10000
        data = {
            'transaction_id': [f'TXN{i}' for i in range(n_samples)],
            'amount': np.random.exponential(1000, n_samples),
            'transaction_type': np.random.choice(['TRANSFER', 'WITHDRAWAL', 'DEPOSIT', 'PAYMENT'], n_samples),
            'concurrent_sessions': np.random.poisson(3, n_samples),
            'tables_locked': np.random.randint(1, 5, n_samples),
            'processing_time_ms': np.random.exponential(500, n_samples),
        }
        
        df = pd.DataFrame(data)
        df['deadlock_occurred'] = self.simulate_deadlocks(df)
        
        return df
    
    def process_user_data(self, df):
        """Process user-uploaded data for prediction"""
        processed_df = df.copy()
        
        # Add missing columns with defaults
        if 'amount' not in processed_df.columns:
            processed_df['amount'] = 1000
        
        if 'transaction_type' not in processed_df.columns:
            if 'TRANSACTION DETAILS' in processed_df.columns:
                processed_df['transaction_type'] = processed_df['TRANSACTION DETAILS'].apply(
                    self.classify_transaction_type
                )
            else:
                processed_df['transaction_type'] = 'OTHER'
        
        # Add ML features
        processed_df = self.create_features(processed_df)
        
        return processed_df
    
    def classify_transaction_type(self, detail):
        """Classify transaction type from details"""
        if pd.isna(detail):
            return 'OTHER'
        
        detail_str = str(detail).upper()
        
        if any(word in detail_str for word in ['TRANSFER', 'NEFT', 'RTGS']):
            return 'TRANSFER'
        elif any(word in detail_str for word in ['WITHDRAWAL', 'ATM', 'CASH']):
            return 'WITHDRAWAL'
        elif any(word in detail_str for word in ['DEPOSIT', 'CREDIT']):
            return 'DEPOSIT'
        elif any(word in detail_str for word in ['CHEQUE', 'CHQ']):
            return 'CHEQUE'
        elif any(word in detail_str for word in ['PAYMENT', 'BILL', 'EMI']):
            return 'PAYMENT'
        else:
            return 'OTHER'