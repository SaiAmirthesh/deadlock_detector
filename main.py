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

from data.data_loader import DataLoader
from utils.conventional_detector import ConventionalDetector
from utils.ml_detector import MLDetector
from utils.hybrid_detector import HybridDetector
from utils.visualization import Visualization

# Page configuration
st.set_page_config(
    page_title="Bank Deadlock Detection System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DeadlockDetectionApp:
    def __init__(self):
        self.data_loader = DataLoader()
        self.conventional_detector = ConventionalDetector()
        self.ml_detector = MLDetector()
        self.hybrid_detector = HybridDetector()
        self.visualization = Visualization()
        
        # Load ML models
        self.ml_detector.load_models()
        
        # Initialize session state
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'user_data' not in st.session_state:
            st.session_state.user_data = None
    
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.header("🏦 Deadlock Detection System")
            
            # File Upload Section
            st.subheader("📁 Upload Your Data")
            uploaded_file = st.file_uploader(
                "Choose Excel/CSV file",
                type=['xlsx', 'xls', 'csv'],
                help="Upload your transaction data"
            )
            
            if uploaded_file is not None:
                self.process_user_upload(uploaded_file)
            
            # Show current data status
            if st.session_state.user_data is not None:
                st.success(f"✅ Data loaded: {len(st.session_state.user_data)} transactions")
            
            # Detection Methods
            st.subheader("🔍 Detection Methods")
            methods = st.multiselect(
                "Select methods to compare:",
                ["Wait-for Graph", "Resource Allocation", "Banker's Algorithm", "Random Forest", "XGBoost", "Hybrid"],
                ["Banker's Algorithm", "Random Forest", "Hybrid"]
            )
            
            # Run Analysis Button
            if st.button("🚀 Run Deadlock Analysis", type="primary", use_container_width=True):
                if st.session_state.user_data is not None:
                    self.run_analysis(methods)
                else:
                    st.error("❌ Please upload data first!")
            
            # Training Section
            st.subheader("🔄 Model Training")
            if st.button("Train ML Models", use_container_width=True):
                self.train_models()
    
    def process_user_upload(self, uploaded_file):
        """Process user-uploaded file"""
        try:
            with st.spinner("Processing your data..."):
                # Read file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Process data
                processed_df = self.data_loader.process_user_data(df)
                st.session_state.user_data = processed_df
                
                st.success(f"✅ Processed {len(df)} transactions")
                
                # Show data preview
                with st.expander("📊 Data Preview"):
                    st.dataframe(processed_df.head())
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    def run_analysis(self, methods):
        """Run deadlock detection analysis"""
        try:
            data = st.session_state.user_data
            results = {}
            
            for method in methods:
                start_time = time.time()
                
                if method in ["Wait-for Graph", "Resource Allocation", "Banker's Algorithm"]:
                    result = self.conventional_detector.detect_deadlock(data, method)
                elif method in ["Random Forest", "XGBoost"]:
                    result = self.ml_detector.detect_deadlock(data, method)
                else:  # Hybrid
                    result = self.hybrid_detector.detect_deadlock(data)
                
                result['time'] = time.time() - start_time
                results[method] = result
            
            st.session_state.results = results
            st.success("✅ Analysis completed!")
            
        except Exception as e:
            st.error(f"❌ Analysis error: {str(e)}")
    
    def train_models(self):
        """Train ML models"""
        try:
            with st.spinner("Training ML models with Kaggle data..."):
                # Import and run training
                from train_models import train_models as train_ml
                train_ml()
                
                # Reload models
                self.ml_detector.load_models()
                st.success("✅ Models trained and loaded successfully!")
                
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
    
    def render_main(self):
        """Render main content"""
        st.title("🏦 Bank Transaction Deadlock Detection")
        st.markdown("Detect potential deadlocks in banking transactions using multiple algorithms")
        
        if st.session_state.results:
            self.render_results()
        else:
            self.render_welcome()
    
    def render_results(self):
        """Render analysis results"""
        results = st.session_state.results
        data = st.session_state.user_data
        
        # Results Overview
        st.header("📊 Detection Results")
        
        # Results cards
        cols = st.columns(len(results))
        for idx, (method, result) in enumerate(results.items()):
            with cols[idx]:
                deadlock = result['deadlock']
                confidence = result.get('confidence', 0)
                
                color = "red" if deadlock else "green"
                icon = "🔴" if deadlock else "🟢"
                
                st.markdown(f"""
                <div style="border: 2px solid {color}; border-radius: 10px; padding: 15px; text-align: center;">
                    <h3>{method}</h3>
                    <h2>{icon} {'Deadlock' if deadlock else 'No Deadlock'}</h2>
                    <p>Confidence: {confidence:.1%}</p>
                    <p>Time: {result.get('time', 0):.3f}s</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed Results
        st.header("🔍 Detailed Analysis")
        
        # Method-specific details
        tabs = st.tabs(list(results.keys()))
        for idx, (method, result) in enumerate(results.items()):
            with tabs[idx]:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader(f"{method} Analysis")
                    if 'error' in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.write(f"**Confidence:** {result.get('confidence', 0):.3f}")
                        st.write(f"**Processing Time:** {result.get('time', 0):.4f}s")
                        
                        if 'method_used' in result:
                            st.write(f"**Method Used:** {result['method_used']}")
                            st.write(f"**Decision Reason:** {result.get('decision_reason', 'N/A')}")
                        
                        if 'factors' in result:
                            st.write("**Key Factors:**")
                            for factor, weight in result['factors'].items():
                                st.write(f"- {factor}: {weight:.3f}")
                
                with col2:
                    st.metric("Deadlock", "Detected" if result['deadlock'] else "Not Detected")
                    st.metric("Confidence", f"{result.get('confidence', 0):.1%}")
        
        # Visualizations
        st.header("📈 Data Analysis")
        
        # Method comparison
        fig_comparison = self.visualization.create_method_comparison(results)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Transaction analysis
        fig_analysis = self.visualization.create_transaction_analysis(data)
        st.plotly_chart(fig_analysis, use_container_width=True)
    
    def render_welcome(self):
        """Render welcome screen"""
        st.markdown("""
        ## 🎯 Welcome to Deadlock Detection System
        
        This system analyzes banking transactions to detect potential database deadlocks using:
        
        - **Traditional Algorithms**: Wait-for Graph, Resource Allocation, Banker's Algorithm
        - **Machine Learning**: Random Forest, XGBoost trained on Kaggle data
        - **Hybrid Approach**: Intelligent method selection
        
        ### 🚀 How to Use:
        1. **Upload** your transaction data (Excel/CSV)
        2. **Select** detection methods to compare
        3. **Run** the analysis
        4. **View** results and insights
        
        ### 📊 Expected Data Format:
        Your file should contain transaction details like:
        - Account information
        - Transaction amounts  
        - Transaction types
        - Timestamps
        
        The system will automatically process your data!
        """)
        
        # Quick stats if models are loaded
        if self.ml_detector.is_trained:
            st.success("✅ ML models are trained and ready!")
        else:
            st.warning("⚠️ ML models not trained. Click 'Train ML Models' in sidebar.")

def main():
    app = DeadlockDetectionApp()
    app.render_sidebar()
    app.render_main()

if __name__ == "__main__":
    main()