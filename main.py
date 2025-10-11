import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))

from data.data_loader import BankDataLoader
from utils.conventional_detector import BankersAlgorithm
from utils.ml_detector import MLDeadlockDetector
from utils.hybrid_detector import HybridDetector
from utils.visualization import ResultsVisualizer

# Page configuration
st.set_page_config(
    page_title="Bank Statement Deadlock Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .deadlock-detected {
        border-left: 4px solid #ff4b4b !important;
        background: #fff5f5;
    }
    .no-deadlock {
        border-left: 4px solid #00d26a !important;
        background: #f0fff4;
    }
    .section-header {
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class BankStatementAnalyzer:
    def __init__(self):
        self.data_loader = BankDataLoader()
        self.bankers = BankersAlgorithm()
        self.ml_detector = MLDeadlockDetector()
        self.hybrid_detector = HybridDetector()
        self.visualizer = ResultsVisualizer()
        
        # Load data
        self.transactions = self.load_data()
        
        # Initialize session state
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'comparison_history' not in st.session_state:
            st.session_state.comparison_history = []
        if 'ml_trained' not in st.session_state:
            st.session_state.ml_trained = False
    
    def load_data(self):
        """Load bank statement data"""
        # Update this path to your actual Excel file
        excel_file_path = "data/bank_statement.xlsx"  # Change this path
        return self.data_loader.load_bank_statement_data(excel_file_path)
    
    def train_ml_models(self):
        """Train ML models on the dataset"""
        with st.spinner("Training ML models on bank statement data..."):
            try:
                results = self.ml_detector.train_models(self.transactions)
                if results:
                    st.session_state.ml_trained = True
                    st.success("ML models trained successfully!")
                    return True
                else:
                    st.error("Failed to train ML models")
                    return False
            except Exception as e:
                st.error(f"Error training ML models: {str(e)}")
                return False
    
    def run_detection(self, transaction_batch, methods):
        """Run deadlock detection with selected methods"""
        results = {}
        
        for method in methods:
            start_time = time.time()
            
            try:
                if method == 'Bankers Algorithm':
                    result = self.bankers.detect_deadlock(transaction_batch)
                elif method == 'Random Forest':
                    result = self.ml_detector.detect_rf(transaction_batch)
                elif method == 'XGBoost':
                    result = self.ml_detector.detect_xgb(transaction_batch)
                elif method == 'Hybrid':
                    result = self.hybrid_detector.detect(transaction_batch)
                else:
                    continue
                
                result['time'] = time.time() - start_time
                results[method] = result
                
            except Exception as e:
                results[method] = {
                    'deadlock': False,
                    'confidence': 0.0,
                    'error': str(e),
                    'time': 0,
                    'method': method
                }
        
        return results
    
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.markdown("## 🏦 Bank Statement Analyzer")
            
            # Data overview
            if hasattr(self, 'transactions') and self.transactions is not None:
                st.metric("Total Transactions", len(self.transactions))
                deadlock_rate = self.transactions['deadlock_occurred'].mean() if 'deadlock_occurred' in self.transactions.columns else 0
                st.metric("Deadlock Rate", f"{deadlock_rate:.1%}")
            else:
                st.metric("Total Transactions", 0)
                st.metric("Deadlock Rate", "0%")
            
            st.markdown("---")
            st.markdown("### 🔧 Analysis Settings")
            
            # Batch size
            batch_size = st.slider("Transaction Batch Size", 10, 100, 30)
            
            # Transaction filters
            st.markdown("#### Transaction Filters")
            
            # Transaction types
            if hasattr(self, 'transactions') and self.transactions is not None and 'transaction_type' in self.transactions.columns:
                available_types = self.transactions['transaction_type'].unique().tolist()
                selected_types = st.multiselect(
                    "Transaction Types",
                    options=available_types,
                    default=available_types[:3] if available_types else []
                )
            else:
                selected_types = []
                st.info("No transaction types available")
            
            # Amount range
            if hasattr(self, 'transactions') and self.transactions is not None and 'amount' in self.transactions.columns:
                max_amount = self.transactions['amount'].max()
                amount_range = st.slider(
                    "Amount Range",
                    0, int(max_amount), (0, int(max_amount * 0.7))
                )
            else:
                amount_range = (0, 10000)
                st.slider("Amount Range", 0, 10000, (0, 5000))
            
            # Method selection
            st.markdown("#### Detection Methods")
            available_methods = ['Bankers Algorithm', 'Random Forest', 'XGBoost', 'Hybrid']
            selected_methods = st.multiselect(
                "Select Methods to Compare",
                options=available_methods,
                default=['Bankers Algorithm', 'Hybrid']  # Default to methods that don't require training
            )
            
            # ML training
            if not st.session_state.ml_trained and any(m in selected_methods for m in ['Random Forest', 'XGBoost']):
                if st.button("🔄 Train ML Models", type="secondary", use_container_width=True):
                    self.train_ml_models()
            
            # Run analysis button
            if st.button("🚀 Analyze for Deadlocks", type="primary", use_container_width=True):
                filters = {
                    'transaction_types': selected_types,
                    'amount_range': amount_range,
                }
                self.run_analysis(batch_size, filters, selected_methods)
            
            st.markdown("---")
            if st.button("🔄 Reset Analysis", type="secondary"):
                st.session_state.results = None
                st.rerun()
    
    def run_analysis(self, batch_size, filters, methods):
        """Run the analysis with given parameters"""
        try:
            # Get transaction batch
            batch = self.data_loader.get_transaction_batch(self.transactions, batch_size, filters)
            
            if len(batch) == 0:
                st.error("No transactions match the selected filters!")
                return
            
            # Run detection
            with st.spinner(f"Analyzing {len(batch)} transactions for deadlocks..."):
                results = self.run_detection(batch, methods)
            
            st.session_state.results = results
            st.session_state.current_batch = batch
            
            # Store for comparison
            st.session_state.comparison_history.append({
                'timestamp': time.time(),
                'batch_size': len(batch),
                'methods': methods,
                'results': results
            })
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
    
    def render_main(self):
        """Render main content"""
        st.markdown('<h1 class="main-header">🏦 Bank Statement Deadlock Analyzer</h1>', unsafe_allow_html=True)
        st.markdown("""
        Analyze your bank statement transactions for potential database deadlocks using advanced detection algorithms.
        """)
        
        if st.session_state.results:
            self.render_quick_results()
            self.render_detailed_analysis()
            self.render_method_comparison()
        else:
            self.render_welcome_screen()
    
    def render_quick_results(self):
        """Render quick overview results"""
        st.markdown('<div class="section-header">📊 Quick Results</div>', unsafe_allow_html=True)
        
        results = st.session_state.results
        batch = st.session_state.current_batch
        
        # Results cards
        cols = st.columns(len(results))
        for idx, (method, result) in enumerate(results.items()):
            with cols[idx]:
                deadlock = result['deadlock']
                confidence = result.get('confidence', 0)
                processing_time = result.get('time', 0)
                
                card_class = "deadlock-detected" if deadlock else "no-deadlock"
                icon = "🔴" if deadlock else "🟢"
                status = "DEADLOCK" if deadlock else "SAFE"
                
                st.markdown(f"""
                <div class="metric-card {card_class}">
                    <h3 style="margin: 0; color: {'#ff4b4b' if deadlock else '#00d26a'}">{method}</h3>
                    <h2 style="margin: 10px 0; font-size: 1.5rem;">{icon} {status}</h2>
                    <p style="margin: 5px 0;"><b>Confidence:</b> {confidence:.1%}</p>
                    <p style="margin: 5px 0;"><b>Time:</b> {processing_time:.3f}s</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Batch statistics
        st.markdown("#### 📈 Batch Statistics")
        stat_cols = st.columns(4)
        
        with stat_cols[0]:
            st.metric("Transactions Analyzed", len(batch))
        with stat_cols[1]:
            total_amount = batch['amount'].sum() if 'amount' in batch.columns else 0
            st.metric("Total Amount", f"${total_amount:,.0f}")
        with stat_cols[2]:
            if 'concurrent_sessions' in batch.columns:
                avg_sessions = batch['concurrent_sessions'].mean()
                st.metric("Avg Sessions", f"{avg_sessions:.1f}")
            else:
                st.metric("Avg Sessions", "N/A")
        with stat_cols[3]:
            if 'deadlock_occurred' in batch.columns:
                actual_deadlocks = batch['deadlock_occurred'].sum()
                st.metric("Deadlock Risk", f"{actual_deadlocks/len(batch):.1%}")
            else:
                st.metric("Deadlock Risk", "N/A")
    
    def render_detailed_analysis(self):
        """Render detailed analysis"""
        st.markdown('<div class="section-header">🔍 Detailed Analysis</div>', unsafe_allow_html=True)
        
        results = st.session_state.results
        batch = st.session_state.current_batch
        
        # Method-specific details in tabs
        if results:
            tabs = st.tabs([f"{method} Details" for method in results.keys()])
            
            for idx, (method, result) in enumerate(results.items()):
                with tabs[idx]:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader(f"{method} Analysis")
                        
                        if 'error' in result:
                            st.error(f"**Error:** {result['error']}")
                        else:
                            # Key metrics
                            st.write(f"**Confidence Score:** {result.get('confidence', 0):.3f}")
                            st.write(f"**Processing Time:** {result.get('time', 0):.4f} seconds")
                            
                            # Hybrid method details
                            if 'method_used' in result:
                                st.write(f"**Method Used:** {result['method_used']}")
                                st.write(f"**Decision Reason:** {result.get('decision_reason', 'N/A')}")
                            
                            # Traditional method details
                            if 'safe_sequence' in result:
                                if result['safe_sequence']:
                                    st.write(f"**Safe Sequence:** {len(result['safe_sequence'])} transactions can proceed safely")
                                else:
                                    st.write("**Safe Sequence:** No safe sequence found")
                            
                            # Key factors
                            if 'factors' in result:
                                st.write("**Key Factors Influencing Detection:**")
                                for factor, weight in result['factors'].items():
                                    st.write(f"- {factor.replace('_', ' ').title()}: {weight:.3f}")
                    
                    with col2:
                        # Quick stats card
                        st.markdown("""
                        <div style="background: #f0f2f6; padding: 1rem; border-radius: 10px;">
                            <h4>Quick Stats</h4>
                        """, unsafe_allow_html=True)
                        
                        st.metric("Deadlock", "Detected" if result['deadlock'] else "Not Detected")
                        st.metric("Confidence", f"{result.get('confidence', 0):.1%}")
                        
                        if 'completion_rate' in result:
                            st.metric("Completion Rate", f"{result['completion_rate']:.1%}")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
        
        # Transaction batch preview
        st.markdown("#### 📋 Transaction Batch Preview")
        
        # Show relevant columns
        display_columns = ['transaction_details', 'transaction_type', 'amount', 
                          'concurrent_sessions', 'tables_locked', 'deadlock_occurred']
        available_columns = [col for col in display_columns if col in batch.columns]
        
        if available_columns:
            st.dataframe(batch[available_columns].head(10), use_container_width=True)
        else:
            st.info("No transaction details available for display")
        
        # Visualization
        st.markdown("#### 📊 Transaction Analysis")
        try:
            fig = self.visualizer.create_transaction_analysis(batch)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate visualization: {e}")
    
    def render_method_comparison(self):
        """Render method comparison"""
        st.markdown('<div class="section-header">⚖️ Method Comparison</div>', unsafe_allow_html=True)
        
        results = st.session_state.results
        
        # Performance comparison chart
        try:
            fig = self.visualizer.create_method_comparison(results)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating comparison chart: {e}")
        
        # Feature importance
        st.markdown("#### 🎯 Key Deadlock Indicators")
        
        # Common deadlock factors
        common_factors = {
            'High Transaction Amount': 0.25,
            'Multiple Concurrent Sessions': 0.22,
            'Complex Transaction Type': 0.18,
            'Multiple Tables Locked': 0.15,
            'Long Processing Time': 0.12,
            'System Load': 0.08
        }
        
        fig_importance = px.bar(
            x=list(common_factors.values()),
            y=list(common_factors.keys()),
            orientation='h',
            title='Common Deadlock Risk Factors',
            labels={'x': 'Importance Weight', 'y': 'Risk Factors'}
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    def render_welcome_screen(self):
        """Render welcome screen"""
        st.markdown("""
        ## Welcome to Bank Statement Deadlock Analyzer
        
        This system analyzes your bank statement transactions to detect potential database deadlocks using:
        
        - **Traditional Method**: Banker's Algorithm for resource allocation analysis
        - **ML Methods**: Random Forest & XGBoost for pattern recognition  
        - **Hybrid Approach**: Intelligent method selection based on data characteristics
        
        ### 🎯 How to Use:
        1. **Configure** transaction filters in the sidebar
        2. **Select** detection methods to compare
        3. **Run** the analysis to detect potential deadlocks
        4. **Review** results and method comparisons
        
        ### 📊 Your Data Overview:
        """)
        
        # Data overview
        if hasattr(self, 'transactions') and self.transactions is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transactions", len(self.transactions))
            with col2:
                if 'account_no' in self.transactions.columns:
                    unique_accounts = self.transactions['account_no'].nunique()
                    st.metric("Unique Accounts", unique_accounts)
                else:
                    st.metric("Unique Accounts", "N/A")
            with col3:
                if 'deadlock_occurred' in self.transactions.columns:
                    deadlock_rate = self.transactions['deadlock_occurred'].mean()
                    st.metric("Overall Deadlock Risk", f"{deadlock_rate:.1%}")
                else:
                    st.metric("Overall Deadlock Risk", "N/A")
            with col4:
                if 'amount' in self.transactions.columns:
                    avg_amount = self.transactions['amount'].mean()
                    st.metric("Avg Amount", f"${avg_amount:,.0f}")
                else:
                    st.metric("Avg Amount", "N/A")
            
            # Sample data preview
            st.markdown("#### Sample Transactions")
            preview_cols = ['account_no', 'date', 'transaction_details', 'amount', 'transaction_type']
            available_preview_cols = [col for col in preview_cols if col in self.transactions.columns]
            
            if available_preview_cols:
                st.dataframe(self.transactions[available_preview_cols].head(8), use_container_width=True)
            else:
                st.info("No transaction data available for preview")
        else:
            st.error("No transaction data loaded. Please check your data file.")

def main():
    try:
        app = BankStatementAnalyzer()
        app.render_sidebar()
        app.render_main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check that all required files are in the correct locations.")

if __name__ == "__main__":
    main()