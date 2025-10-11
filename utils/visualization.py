import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

class ResultsVisualizer:
    def __init__(self):
        self.colors = {
            'Bankers Algorithm': '#1f77b4',
            'Random Forest': '#ff7f0e', 
            'XGBoost': '#2ca02c',
            'Hybrid': '#d62728'
        }
    
    def create_method_comparison(self, results):
        """Create comparison chart of all methods"""
        methods = list(results.keys())
        accuracies = [results[m]['confidence'] for m in methods]
        times = [results[m]['time'] for m in methods]
        deadlocks = [1 if results[m]['deadlock'] else 0 for m in methods]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confidence Scores', 'Processing Time (s)', 'Deadlock Detection', 'Overall Performance'),
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Confidence scores
        fig.add_trace(
            go.Bar(x=methods, y=accuracies, name='Confidence', marker_color=[self.colors[m] for m in methods]),
            row=1, col=1
        )
        
        # Processing time
        fig.add_trace(
            go.Bar(x=methods, y=times, name='Time (s)', marker_color=[self.colors[m] for m in methods]),
            row=1, col=2
        )
        
        # Deadlock detection
        fig.add_trace(
            go.Bar(x=methods, y=deadlocks, name='Deadlock', marker_color=[self.colors[m] for m in methods]),
            row=2, col=1
        )
        
        # Overall performance (composite score)
        performance_scores = [
            (acc - (time * 0.1) + (deadlock * 0.2)) for acc, time, deadlock in zip(accuracies, times, deadlocks)
        ]
        fig.add_trace(
            go.Scatter(x=methods, y=performance_scores, name='Performance', 
                      marker=dict(size=15, color=[self.colors[m] for m in methods])),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Method Comparison Analysis")
        return fig
    
    def create_feature_importance(self, feature_weights):
        """Create feature importance visualization"""
        features = list(feature_weights.keys())
        importance = list(feature_weights.values())
        
        fig = px.bar(
            x=importance, y=features, orientation='h',
            title='Key Deadlock Indicators',
            labels={'x': 'Importance', 'y': 'Features'}
        )
        fig.update_layout(height=400)
        return fig
    
    def create_transaction_analysis(self, transactions):
        """Create transaction batch analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Transaction Types', 'Amount Distribution', 'Concurrent Sessions', 'Processing Time'),
            specs=[[{"type": "pie"}, {"type": "histogram"}], [{"type": "histogram"}, {"type": "histogram"}]]
        )
        
        # Transaction types
        type_counts = transactions['transaction_type'].value_counts()
        fig.add_trace(
            go.Pie(labels=type_counts.index, values=type_counts.values, name='Types'),
            row=1, col=1
        )
        
        # Amount distribution
        fig.add_trace(
            go.Histogram(x=transactions['amount'], name='Amounts', nbinsx=20),
            row=1, col=2
        )
        
        # Concurrent sessions
        fig.add_trace(
            go.Histogram(x=transactions['concurrent_sessions'], name='Sessions', nbinsx=10),
            row=2, col=1
        )
        
        # Processing time
        fig.add_trace(
            go.Histogram(x=transactions['processing_time_ms'], name='Processing Time', nbinsx=20),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Transaction Batch Analysis")
        return fig