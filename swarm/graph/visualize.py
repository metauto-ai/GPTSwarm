#!/usr/bin/env python
# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import seaborn as sns

def GPTSwarmVis(graph, style="pyvis", dry_run: bool = False, file_name=None):
    G = nx.DiGraph()
    edge_labels = {}  
    order_counter = 0  

    for node_id, node in graph.nodes.items():
        G.add_node(node_id, label=f"{type(node).__name__}\n(ID: {node_id})")
    for node_id, node in graph.nodes.items():
        for successor in node.successors:
            G.add_edge(node_id, successor.id)
            edge_labels[(node_id, successor.id)] = f"Order: {order_counter}"
            order_counter += 1

    if style == "pyvis":
        if hasattr(graph, 'graphs'):
            color_map = generate_color_map([g.id for g in graph.graphs] + [''])
        else:
            color_map = generate_color_map(graph.nodes.keys())

        net = Network(notebook=True, height="750px", width="100%", bgcolor="#FFFFFF", font_color="black", directed=True)
        for node_id, node in graph.nodes.items():
            if hasattr(graph, 'graphs'):
                graph_id = ''
                for g in graph.graphs:
                    if node_id in g.nodes:
                        graph_id = g.id
                        break
                    color_key = graph_id
            else:
                color_key = node_id
            net.add_node(node_id, label=f"{type(node).__name__}\n(ID: {node_id})", color=color_map[color_key])

        for node_id, node in graph.nodes.items():
            for successor in node.successors:
                net.add_edge(node_id, successor.id)

        if not dry_run:
            import os
            from swarm.utils.const import GPTSWARM_ROOT
            result_path = GPTSWARM_ROOT / "result"
            os.makedirs(result_path, exist_ok=True)
            net.show(f"{result_path}/{file_name if file_name else 'example.html'}")
            os.system(f"open {GPTSWARM_ROOT}/result/{file_name if file_name else 'example.html'}")

    else:
        pos = nx.spring_layout(G, k=.3, iterations=30)
        node_colors = [color_map[node] for node in G.nodes()]
        node_sizes = [3000 + 100 * G.degree[node] for node in G.nodes()]
        plt.figure(figsize=(12, 12))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=0.93)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.6, node_shape='o', edgecolors='black', linewidths=1.5)
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='-|>', node_size=node_sizes, arrowsize=20, edge_color='grey')
        nx.draw_networkx_edge_labels(G, pos, font_size=7, edge_labels=edge_labels, font_color='red')

        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7, font_family='sans-serif', font_weight='bold', font_color='blue')
        
        plt.title(f"GPTSwarm", size=20, color='darkblue', fontweight='bold', fontfamily='sans-serif')
        plt.axis('off')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        if not dry_run:
            plt.show()

def generate_color_map(node_ids):
    color_palette = sns.color_palette("husl", len(node_ids)).as_hex()
    color_map = {node_id: color_palette[i % len(color_palette)] for i, node_id in enumerate(node_ids)}
    return color_map

# Example usage
# test_graph = {
#     'A': Node("Start", successors=['B']),
#     'B': Node("Process", predecessors=['A'], successors=['C', 'D']),
#     # ... other nodes ...
# }
# GPTSwarmVis(test_graph, style="html")
