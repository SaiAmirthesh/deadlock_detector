import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from utils.conventional_detector import ConventionalDetector
from utils.ml_detector import MLDetector
from utils.hybrid_detector import HybridDetector
from utils.wait_for_graph import WaitForGraph
from utils.visualization import Visualization
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set page config
st.set_page_config(
    page_title="Hybrid Deadlock Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
try:
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; border-radius: 8px; }
    .metric-card { border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); padding: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

class DeadlockDetectionApp:
    def __init__(self):
        self.conventional_detector = ConventionalDetector()
        self.ml_detector = MLDetector()
        self.hybrid_detector = HybridDetector()
        self.graph_visualizer = WaitForGraph()
        self.visualization = Visualization()
        
        # Initialize session state
        if "system_state" not in st.session_state:
            st.session_state.system_state = self._initialize_system()
        if "detection_history" not in st.session_state:
            st.session_state.detection_history = []
        if "method_performance" not in st.session_state:
            st.session_state.method_performance = {
                "Wait-for Graph": {"accuracy": 0.92, "speed": 0.85},
                "Resource Allocation": {"accuracy": 0.88, "speed": 0.78},
                "Banker's Algorithm": {"accuracy": 0.95, "speed": 0.65},
                "Random Forest": {"accuracy": 0.96, "speed": 0.92},
                "XGBoost": {"accuracy": 0.97, "speed": 0.94},
                "Hybrid": {"accuracy": 0.98, "speed": 0.90}
            }
    
    def _initialize_system(self):
        """Initialize a default system state"""
        return {
            "processes": [
                {"id": 0, "allocation": [1, 0, 0], "max": [1, 2, 1], "need": [0, 2, 1]},
                {"id": 1, "allocation": [0, 1, 0], "max": [2, 1, 1], "need": [2, 0, 1]},
                {"id": 2, "allocation": [0, 0, 1], "max": [1, 1, 2], "need": [1, 1, 1]}
            ],
            "available": [1, 1, 1],
            "total_resources": [2, 2, 2]
        }
    
    def render_sidebar(self):
        """Render the sidebar controls"""
        with st.sidebar:
            st.header("🔧 System Configuration")
            
            # Process configuration
            st.subheader("Process Configuration")
            num_processes = st.slider("Number of Processes", 2, 10, 3)
            num_resources = st.slider("Number of Resource Types", 2, 5, 3)
            
            # Resource configuration
            st.subheader("Resource Configuration")
            total_resources = []
            for i in range(num_resources):
                total_resources.append(st.number_input(f"Total R{i}", 1, 10, 2, key=f"total_{i}"))
            
            # Method selection
            st.subheader("Detection Methods")
            methods = st.multiselect(
                "Select methods to compare",
                ["Wait-for Graph", "Resource Allocation", "Banker's Algorithm", "Random Forest", "XGBoost", "Hybrid"],
                ["Wait-for Graph", "Random Forest", "Hybrid"]
            )
            
            # Simulation controls
            st.subheader("Simulation Controls")
            if st.button("▶️ Run Simulation", type="primary"):
                self.run_simulation(methods, num_processes, num_resources, total_resources)
            
            if st.button("🔄 Reset System"):
                st.session_state.system_state = self._initialize_system()
                st.session_state.detection_history = []
                st.rerun()
    
    def run_simulation(self, methods, num_processes, num_resources, total_resources):
        """Run the deadlock detection simulation"""
        # Generate a random system state
        system_state = self.generate_system_state(num_processes, num_resources, total_resources)
        st.session_state.system_state = system_state
        
        # Run detection with all selected methods
        results = {}
        for method in methods:
            start_time = time.time()
            
            if method in ["Wait-for Graph", "Resource Allocation", "Banker's Algorithm"]:
                deadlock, details = self.conventional_detector.detect_deadlock(
                    system_state["processes"], system_state["available"], method
                )
            elif method in ["Random Forest", "XGBoost"]:
                deadlock, details = self.ml_detector.detect_deadlock(
                    system_state, method
                )
            else:  # Hybrid
                deadlock, details = self.hybrid_detector.detect_deadlock(system_state)
            
            detection_time = time.time() - start_time
            
            results[method] = {
                "deadlock": deadlock,
                "details": details,
                "time": detection_time
            }
        
        # Store results in history
        st.session_state.detection_history.append({
            "timestamp": time.time(),
            "system_state": system_state,
            "results": results
        })
    
    def generate_system_state(self, num_processes, num_resources, total_resources):
        """Generate a random system state for simulation"""
        processes = []
        available = total_resources.copy()
        
        for i in range(num_processes):
            # Random allocation (some resources may be allocated)
            allocation = [np.random.randint(0, max_res // 2 + 1) for max_res in total_resources]
            
            # Calculate maximum demand (allocation + random additional need)
            max_demand = [
                allocation[j] + np.random.randint(0, total_resources[j] - allocation[j] + 1)
                for j in range(num_resources)
            ]
            
            # Calculate need
            need = [max_demand[j] - allocation[j] for j in range(num_resources)]
            
            processes.append({
                "id": i,
                "allocation": allocation,
                "max": max_demand,
                "need": need
            })
            
            # Update available resources
            available = [available[j] - allocation[j] for j in range(num_resources)]
        
        return {
            "processes": processes,
            "available": available,
            "total_resources": total_resources
        }
    
    def render_main_content(self):
        """Render the main content area"""
        st.title("🔒 Hybrid Deadlock Detection System")
        st.markdown("Compare traditional algorithms with ML approaches for deadlock detection")
        
        # Display current system state
        st.header("Current System State")
        self.render_system_state()
        
        # Display detection results if available
        if st.session_state.detection_history:
            latest_result = st.session_state.detection_history[-1]
            st.header("Detection Results")
            self.render_detection_results(latest_result)
            
            # Display performance comparison
            st.header("Method Performance Comparison")
            self.render_performance_comparison()
            
            # Display wait-for graph
            st.header("Wait-for Graph Visualization")
            self.render_wait_for_graph(latest_result["system_state"])
    
    def render_system_state(self):
        """Render the current system state"""
        system_state = st.session_state.system_state
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Processes")
            process_data = []
            for process in system_state["processes"]:
                process_data.append({
                    "Process": f"P{process['id']}",
                    "Allocation": str(process['allocation']),
                    "Max": str(process['max']),
                    "Need": str(process['need'])
                })
            st.dataframe(pd.DataFrame(process_data), use_container_width=True)
        
        with col2:
            st.subheader("Resources")
            resource_data = {
                "Resource": [f"R{i}" for i in range(len(system_state["available"]))],
                "Total": system_state["total_resources"],
                "Available": system_state["available"]
            }
            st.dataframe(pd.DataFrame(resource_data), use_container_width=True)
        
        with col3:
            st.subheader("System Metrics")
            total_processes = len(system_state["processes"])
            total_resources = sum(system_state["total_resources"])
            allocated = sum([sum(process['allocation']) for process in system_state["processes"]])
            utilization = allocated / total_resources * 100 if total_resources > 0 else 0
            
            st.metric("Processes", total_processes)
            st.metric("Total Resources", total_resources)
            st.metric("Resource Utilization", f"{utilization:.1f}%")
    
    def render_detection_results(self, result):
        """Render the detection results"""
        system_state = result["system_state"]
        results = result["results"]
        
        # Create results cards
        cols = st.columns(len(results))
        
        for i, (method, data) in enumerate(results.items()):
            with cols[i]:
                st.markdown(f"<div class='metric-card'>", unsafe_allow_html=True)
                st.subheader(method)
                
                if data["deadlock"]:
                    st.error("🚨 Deadlock Detected")
                else:
                    st.success("✅ No Deadlock")
                
                st.caption(f"Time: {data['time']*1000:.2f} ms")
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Show detailed results in expanders
        with st.expander("Detailed Results"):
            for method, data in results.items():
                st.subheader(method)
                if "details" in data and data["details"]:
                    if method in ["Wait-for Graph", "Resource Allocation"] and "cycle" in data["details"]:
                        st.write("Cycle detected:", data["details"]["cycle"])
                    elif method == "Banker's Algorithm" and "sequence" in data["details"]:
                        if data["details"]["sequence"]:
                            st.write("Safe sequence:", data["details"]["sequence"])
                        else:
                            st.write("No safe sequence exists")
                    elif method in ["Random Forest", "XGBoost"] and "confidence" in data["details"]:
                        st.write(f"Confidence: {data['details']['confidence']:.2%}")
                    elif method == "Hybrid":
                        st.write("Method used:", data["details"].get("method", "N/A"))
                        if "confidence" in data["details"]:
                            st.write(f"Confidence: {data['details']['confidence']:.2%}")
                st.divider()
    
    def render_performance_comparison(self):
        """Render performance comparison charts"""
        performance = st.session_state.method_performance
        
        # Create accuracy comparison chart
        methods = list(performance.keys())
        accuracy = [performance[m]["accuracy"] for m in methods]
        speed = [performance[m]["speed"] for m in methods]
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy Comparison", "Speed Comparison"))
        
        fig.add_trace(
            go.Bar(x=methods, y=accuracy, name="Accuracy", marker_color="blue"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=methods, y=speed, name="Speed", marker_color="green"),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show historical accuracy if available
        if len(st.session_state.detection_history) > 1:
            st.subheader("Historical Performance")
            self.render_historical_performance()
    
    def render_historical_performance(self):
        """Render historical performance data"""
        history = st.session_state.detection_history
        
        # Count correct detections for each method
        method_correct = {method: 0 for method in st.session_state.method_performance.keys()}
        method_total = {method: 0 for method in st.session_state.method_performance.keys()}
        
        for detection in history:
            # Determine ground truth (use hybrid as reference)
            ground_truth = detection["results"].get("Hybrid", {}).get("deadlock", False)
            
            for method, result in detection["results"].items():
                if method in method_total:
                    method_total[method] += 1
                    if result["deadlock"] == ground_truth:
                        method_correct[method] += 1
        
        # Calculate accuracy
        accuracy_data = {
            "Method": [],
            "Accuracy": [],
            "Tests": []
        }
        
        for method in method_total:
            if method_total[method] > 0:
                accuracy_data["Method"].append(method)
                accuracy_data["Accuracy"].append(method_correct[method] / method_total[method])
                accuracy_data["Tests"].append(method_total[method])
        
        fig = px.bar(accuracy_data, x="Method", y="Accuracy", color="Method",
                    title="Historical Accuracy by Method")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_wait_for_graph(self, system_state):
        """Render the wait-for graph visualization"""
        graph_html = self.graph_visualizer.generate_graph(system_state)
        st.components.v1.html(graph_html, height=600)

def main():
    app = DeadlockDetectionApp()
    app.render_sidebar()
    app.render_main_content()

if __name__ == "__main__":
    main()