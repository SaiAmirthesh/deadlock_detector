import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

class Visualization:
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
    
    def create_method_comparison(self, results):
        """Create comparison chart of all methods"""
        methods = list(results.keys())
        accuracies = [results[m].get('confidence', 0) for m in methods]
        times = [results[m].get('time', 0) for m in methods]
        deadlocks = [1 if results[m].get('deadlock', False) else 0 for m in methods]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confidence Scores', 'Processing Time (s)', 'Deadlock Detection', 'Performance Score'),
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Confidence scores
        fig.add_trace(
            go.Bar(x=methods, y=accuracies, name='Confidence', marker_color='blue'),
            row=1, col=1
        )
        
        # Processing time
        fig.add_trace(
            go.Bar(x=methods, y=times, name='Time (s)', marker_color='green'),
            row=1, col=2
        )
        
        # Deadlock detection
        fig.add_trace(
            go.Bar(x=methods, y=deadlocks, name='Deadlock', marker_color='red'),
            row=2, col=1
        )
        
        # Performance score (composite)
        performance_scores = [acc - (time * 2) for acc, time in zip(accuracies, times)]
        fig.add_trace(
            go.Bar(x=methods, y=performance_scores, name='Performance', marker_color='purple'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Method Comparison Analysis")
        return fig
    
    def create_transaction_analysis(self, transactions):
        """Create transaction analysis charts"""
        if 'transaction_type' not in transactions.columns:
            return self._create_default_chart()
        
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
        
        fig.update_layout(height=600, showlegend=False, title_text="Transaction Analysis")
        return fig
    
    def _create_default_chart(self):
        """Create a default chart when data is insufficient"""
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for visualization",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, xanchor='center', yanchor='middle',
                          showarrow=False)
        fig.update_layout(height=400, title_text="Transaction Analysis")
        return fig