import pandas as pd
import numpy as np
import os
from datetime import datetime

class BankDataLoader:
    def __init__(self):
        self.data_path = "data/"
        
    def load_bank_statement_data(self, file_path):
        """Load and process your bank statement Excel data"""
        try:
            # Load Excel file
            df = pd.read_excel(file_path)
            print(f"Loaded bank statement with shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            return self._process_bank_statement(df)
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return self._generate_sample_data()
    
    def _process_bank_statement(self, df):
        """Process your specific bank statement format"""
        processed_df = df.copy()
        
        # Clean column names
        processed_df.columns = [col.strip().lower().replace(' ', '_') for col in processed_df.columns]
        print(f"Cleaned columns: {processed_df.columns.tolist()}")
        
        # Handle missing values
        processed_df.fillna(0, inplace=True)
        
        # Convert date columns
        date_columns = ['date', 'value_date']
        for col in date_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
        
        # Add features for deadlock detection
        processed_df = self._add_deadlock_features(processed_df)
        
        return processed_df
    
    def _add_deadlock_features(self, df):
        """Add features needed for deadlock detection from bank statement"""
        
        # Calculate transaction amounts (handle both withdrawal and deposit columns)
        if 'withdrawal_amt' in df.columns and 'deposit_amt' in df.columns:
            df['amount'] = df['withdrawal_amt'] + df['deposit_amt']
        else:
            df['amount'] = np.random.exponential(1000, len(df))
        
        # Extract transaction type from transaction details
        df['transaction_type'] = df['transaction_details'].apply(
            lambda x: self._classify_transaction_type(str(x))
        )
        
        # Simulate concurrent sessions based on transaction patterns
        df['concurrent_sessions'] = self._simulate_concurrent_sessions(df)
        
        # Simulate tables locked based on transaction complexity
        df['tables_locked'] = self._simulate_tables_locked(df)
        
        # Simulate processing time
        df['processing_time_ms'] = np.random.exponential(500, len(df))
        
        # Simulate deadlock occurrences
        df['deadlock_occurred'] = self._simulate_deadlocks(df)
        
        print(f"Added deadlock features. Deadlock rate: {df['deadlock_occurred'].mean():.2%}")
        return df
    
    def _classify_transaction_type(self, transaction_detail):
        """Classify transaction type from transaction details"""
        detail = str(transaction_detail).upper()
        
        if any(word in detail for word in ['TRANSFER', 'NEFT', 'RTGS', 'IMPS']):
            return 'TRANSFER'
        elif any(word in detail for word in ['WITHDRAWAL', 'ATM', 'CASH']):
            return 'WITHDRAWAL'
        elif any(word in detail for word in ['DEPOSIT', 'CREDIT']):
            return 'DEPOSIT'
        elif any(word in detail for word in ['CHEQUE', 'CHQ']):
            return 'CHEQUE'
        elif any(word in detail for word in ['PAYMENT', 'BILL', 'EMI']):
            return 'PAYMENT'
        else:
            return 'OTHER'
    
    def _simulate_concurrent_sessions(self, df):
        """Simulate concurrent database sessions based on transaction patterns"""
        sessions = []
        
        for _, row in df.iterrows():
            base_sessions = 1
            
            # More sessions for high-value transactions
            amount = row.get('amount', 0)
            if amount > 10000:
                base_sessions += 2
            elif amount > 5000:
                base_sessions += 1
            
            # More sessions for complex transaction types
            tx_type = row.get('transaction_type', 'OTHER')
            if tx_type in ['TRANSFER', 'CHEQUE']:
                base_sessions += 1
            
            # Add some randomness
            sessions.append(max(1, np.random.poisson(base_sessions)))
        
        return sessions
    
    def _simulate_tables_locked(self, df):
        """Simulate number of database tables locked"""
        tables = []
        
        for _, row in df.iterrows():
            base_tables = 1  # Account table always locked
            
            # Additional tables based on transaction type
            tx_type = row.get('transaction_type', 'OTHER')
            if tx_type == 'TRANSFER':
                base_tables += 2  # Source and destination accounts
            elif tx_type == 'CHEQUE':
                base_tables += 1  # Cheque clearing table
            elif tx_type == 'PAYMENT':
                base_tables += 1  # Biller table
            
            # High amount transactions might lock audit tables
            if row.get('amount', 0) > 5000:
                base_tables += 1
            
            tables.append(base_tables)
        
        return tables
    
    def _simulate_deadlocks(self, df):
        """Simulate deadlock occurrences based on real transaction patterns"""
        deadlock_probs = []
        
        for _, row in df.iterrows():
            base_prob = 0.03  # Base 3% chance
            
            # Factors increasing deadlock probability
            amount = row.get('amount', 0)
            if amount > 10000:
                base_prob += 0.12
            elif amount > 5000:
                base_prob += 0.08
            
            # Transaction type factors
            tx_type = row.get('transaction_type', 'OTHER')
            if tx_type == 'TRANSFER':
                base_prob += 0.10  # Involves multiple accounts
            elif tx_type == 'CHEQUE':
                base_prob += 0.06  # Complex clearing process
            
            # Concurrent sessions factor
            sessions = row.get('concurrent_sessions', 1)
            base_prob += min(sessions * 0.02, 0.1)
            
            # Tables locked factor
            tables = row.get('tables_locked', 1)
            base_prob += min(tables * 0.03, 0.15)
            
            deadlock_probs.append(1 if np.random.random() < min(base_prob, 0.4) else 0)
        
        return deadlock_probs
    
    def _generate_sample_data(self):
        """Generate sample data matching your format if Excel file not available"""
        print("Generating sample bank statement data...")
        
        n_samples = 3000
        data = {
            'account_no': [f'ACC{10000 + i}' for i in range(n_samples)],
            'date': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
            'transaction_details': np.random.choice([
                'NEFT TRANSFER', 'ATM WITHDRAWAL', 'CASH DEPOSIT', 
                'CHEQUE CLEARING', 'BILL PAYMENT', 'SALARY CREDIT'
            ], n_samples),
            'chq_no': [''] * n_samples,
            'value_date': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
            'withdrawal_amt': np.random.exponential(500, n_samples),
            'deposit_amt': np.random.exponential(1000, n_samples),
            'balance_amt': np.random.uniform(1000, 100000, n_samples)
        }
        
        df = pd.DataFrame(data)
        df = self._add_deadlock_features(df)
        
        return df
    
    def get_transaction_batch(self, df, batch_size=50, filters=None):
        """Get a batch of transactions for analysis"""
        if filters:
            filtered_df = df.copy()
            
            if 'transaction_types' in filters and filters['transaction_types']:
                filtered_df = filtered_df[filtered_df['transaction_type'].isin(filters['transaction_types'])]
            
            if 'amount_range' in filters:
                min_amt, max_amt = filters['amount_range']
                filtered_df = filtered_df[(filtered_df['amount'] >= min_amt) & (filtered_df['amount'] <= max_amt)]
            
            if 'date_range' in filters and filters['date_range']:
                start_date, end_date = filters['date_range']
                if 'date' in filtered_df.columns:
                    # Convert Python date to pandas Timestamp for comparison
                    start_date = pd.Timestamp(start_date)
                    end_date = pd.Timestamp(end_date)
                    filtered_df = filtered_df[
                        (filtered_df['date'] >= start_date) & 
                        (filtered_df['date'] <= end_date)
                    ]
        else:
            filtered_df = df
        
        if len(filtered_df) < batch_size:
            return filtered_df
        
        return filtered_df.sample(batch_size, random_state=42)