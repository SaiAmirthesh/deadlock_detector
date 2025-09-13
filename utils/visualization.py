import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class Visualization:
    def __init__(self):
        self.colors = px.colors.qualitative.Plotly
    
    def create_performance_comparison(self, performance_data: Dict) -> go.Figure:
        """Create performance comparison charts"""
        methods = list(performance_data.keys())
        accuracy = [performance_data[m].get("accuracy", 0) for m in methods]
        speed = [performance_data[m].get("speed", 0) for m in methods]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Accuracy Comparison", "Speed Comparison"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        fig.add_trace(
            go.Bar(x=methods, y=accuracy, name="Accuracy", marker_color=self.colors[0]),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=methods, y=speed, name="Speed", marker_color=self.colors[1]),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="Method Performance Comparison"
        )
        
        fig.update_yaxes(range=[0, 1], row=1, col=1)
        fig.update_yaxes(range=[0, 1], row=1, col=2)
        
        return fig
    
    def create_historical_accuracy(self, detection_history: List) -> go.Figure:
        """Create historical accuracy chart"""
        if not detection_history:
            return go.Figure()
        
        # Extract accuracy data
        methods = set()
        accuracy_data = {}
        
        for detection in detection_history:
            for method, result in detection["results"].items():
                if method not in methods:
                    methods.add(method)
                    accuracy_data[method] = {"correct": 0, "total": 0}
                
                # Use hybrid as ground truth
                ground_truth = detection["results"].get("Hybrid", {}).get("deadlock", False)
                if result["deadlock"] == ground_truth:
                    accuracy_data[method]["correct"] += 1
                accuracy_data[method]["total"] += 1
        
        # Calculate accuracy
        methods = list(methods)
        accuracy = [accuracy_data[m]["correct"] / accuracy_data[m]["total"] if accuracy_data[m]["total"] > 0 else 0 
                   for m in methods]
        tests = [accuracy_data[m]["total"] for m in methods]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=methods,
            y=accuracy,
            text=[f"{acc:.2%}<br>({test} tests)" for acc, test in zip(accuracy, tests)],
            textposition='auto',
            marker_color=self.colors
        ))
        
        fig.update_layout(
            title="Historical Accuracy by Method",
            yaxis_title="Accuracy",
            yaxis_tickformat=".0%",
            yaxis_range=[0, 1]
        )
        
        return fig
    
    def create_resource_utilization(self, system_state: Dict) -> go.Figure:
        """Create resource utilization chart"""
        processes = system_state["processes"]
        available = system_state["available"]
        total_resources = system_state["total_resources"]
        
        # Calculate utilization
        resource_names = [f"R{i}" for i in range(len(available))]
        allocated = [0] * len(available)
        
        for process in processes:
            for j, alloc in enumerate(process["allocation"]):
                allocated[j] += alloc
        
        utilization = [alloc / total for alloc, total in zip(allocated, total_resources)]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=resource_names,
            y=utilization,
            text=[f"{u:.0%}" for u in utilization],
            textposition="auto",
            marker_color=self.colors
        ))
        
        fig.update_layout(
            title="Resource Utilization",
            yaxis_title="Utilization",
            yaxis_tickformat=".0%",
            yaxis_range=[0, 1]
        )
        
        return fig
    
    def create_detection_time_comparison(self, detection_history: List) -> go.Figure:
        """Create detection time comparison chart"""
        if not detection_history:
            return go.Figure()
        
        # Extract time data from latest detection
        latest = detection_history[-1]
        methods = list(latest["results"].keys())
        times = [latest["results"][m]["time"] * 1000 for m in methods]  # Convert to ms
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=methods,
            y=times,
            text=[f"{t:.2f} ms" for t in times],
            textposition="auto",
            marker_color=self.colors
        ))
        
        fig.update_layout(
            title="Detection Time Comparison",
            yaxis_title="Time (ms)"
        )
        
        return fig