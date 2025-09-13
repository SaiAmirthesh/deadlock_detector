import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class WaitForGraph:
    def __init__(self):
        self.colors = {
            "process": "lightblue",
            "resource": "lightgreen",
            "edge": "black",
            "cycle_edge": "red"
        }
    
    def generate_graph(self, system_state: Dict) -> str:
        """
        Generate an HTML representation of the wait-for graph
        """
        processes = system_state["processes"]
        available = system_state["available"]
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add process nodes
        for process in processes:
            G.add_node(f"P{process['id']}", type="process", label=f"P{process['id']}")
        
        # Add resource nodes
        for j in range(len(available)):
            G.add_node(f"R{j}", type="resource", label=f"R{j}\n(Avail: {available[j]})")
        
        # Add edges based on allocation and need
        for process in processes:
            for j in range(len(available)):
                # Allocation edges (resource -> process)
                if process["allocation"][j] > 0:
                    G.add_edge(f"R{j}", f"P{process['id']}", 
                              label=f"Alloc: {process['allocation'][j]}")
                
                # Request edges (process -> resource)
                if process["need"][j] > 0 and available[j] < process["need"][j]:
                    G.add_edge(f"P{process['id']}", f"R{j}", 
                              label=f"Need: {process['need'][j]}")
        
        # Check for cycles
        try:
            cycle = nx.find_cycle(G)
            has_cycle = True
        except nx.NetworkXNoCycle:
            cycle = []
            has_cycle = False
        
        # Generate HTML with Graphviz via Networkx
        try:
            # Use pygraphviz if available
            import pygraphviz as pgv
            A = nx.nx_agraph.to_agraph(G)
            
            # Set node attributes
            for node in A.nodes():
                if node.attr.get("type") == "process":
                    node.attr["shape"] = "ellipse"
                    node.attr["style"] = "filled"
                    node.attr["fillcolor"] = self.colors["process"]
                else:
                    node.attr["shape"] = "box"
                    node.attr["style"] = "filled"
                    node.attr["fillcolor"] = self.colors["resource"]
            
            # Highlight cycle edges
            if has_cycle:
                for i in range(len(cycle)):
                    edge = (cycle[i][0], cycle[i][1])
                    if A.has_edge(*edge):
                        A.get_edge(*edge).attr["color"] = self.colors["cycle_edge"]
                        A.get_edge(*edge).attr["penwidth"] = "2.0"
            
            # Generate graph
            A.layout(prog="dot")
            graph_html = A.draw(format="svg").decode("utf-8")
            
        except ImportError:
            # Fallback to matplotlib
            plt.figure(figsize=(10, 8))
            
            # Create layout
            pos = nx.spring_layout(G)
            
            # Draw nodes
            process_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "process"]
            resource_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "resource"]
            
            nx.draw_networkx_nodes(G, pos, nodelist=process_nodes, 
                                  node_color=self.colors["process"], node_size=2000)
            nx.draw_networkx_nodes(G, pos, nodelist=resource_nodes, 
                                  node_color=self.colors["resource"], node_size=2000)
            
            # Draw edges
            if has_cycle:
                cycle_edges = [(cycle[i][0], cycle[i][1]) for i in range(len(cycle))]
                other_edges = [edge for edge in G.edges if edge not in cycle_edges]
                
                nx.draw_networkx_edges(G, pos, edgelist=other_edges, 
                                      edge_color=self.colors["edge"])
                nx.draw_networkx_edges(G, pos, edgelist=cycle_edges, 
                                      edge_color=self.colors["cycle_edge"], width=2.0)
            else:
                nx.draw_networkx_edges(G, pos, edge_color=self.colors["edge"])
            
            # Draw labels
            labels = {n: G.nodes[n].get("label", n) for n in G.nodes}
            nx.draw_networkx_labels(G, pos, labels)
            
            # Edge labels
            edge_labels = {(u, v): G.edges[u, v].get("label", "") for u, v in G.edges}
            nx.draw_networkx_edge_labels(G, pos, edge_labels)
            
            # Convert to HTML
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            graph_html = f'<img src="data:image/png;base64,{img_str}" width="100%">'
            plt.close()
        
        # Add title and info
        title = f"<h3>Wait-for Graph {'(Deadlock Detected!)' if has_cycle else '(No Deadlock)'}</h3>"
        info = f"<p>Processes: {len(processes)}, Resources: {len(available)}, Cycle: {has_cycle}</p>"
        
        return title + info + graph_html