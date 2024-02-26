import random
from typing import Optional

from swarm.graph import Graph
from swarm.graph.node import Node


class CompositeGraph(Graph):
    """
    The composite graph is a graph that contains other agents as sub-graphs.
    """

    def __init__(self,
                 decision_method: Node,
                 domain: str,
                 model_name: Optional[str] = None,
                 ):
        super().__init__(domain, model_name)
        self.decision_method = decision_method
        self.domain = domain
        self.model_name = model_name
        self.graphs = []
        self.output_nodes = [decision_method]
        self.add_node(self.decision_method)
        

    def add_graph(self, graph):

        for node in graph.nodes.values():
            # We move the check cycle to parameterization.py
            # if self.check_cycle(node):
            #     #raise Exception(f"Adding node {node.id} would cause a cyclic dependency.")
            self.add_node(node)

        self.graphs.append(graph)
        graph.memory = self.memory
        self.input_nodes.extend(graph.input_nodes)

    def build_graph(self):
        pass
        # for decision_node in self.decision_nodes:
        #     for output_node in self.output_nodes:
        #         output_node.add_successor(decision_node)
    
    def init(self, init_connection_probability, potential_connections):
        self.learned_connections = []
        for connection in potential_connections:
            out_node, in_node = connection
            out_node = self.nodes[out_node]
            in_node = self.nodes[in_node]
            if random.random() < init_connection_probability and not self.check_cycle(in_node, {out_node}, set()):
                self.learned_connections.append(connection)
                out_node.add_successor(in_node)

    def mutate(self, max_new_edges, max_remove_edges, potential_connections):
        # Add new edges
        num_new_edges = random.randint(0, max_new_edges)
        num_remove_edges = random.randint(1 if num_new_edges == 0 else 0, max_remove_edges)
        new_edge_count = 0
        for _ in range(num_new_edges * 10):
            connection = random.choice(potential_connections)
            out_node, in_node = connection
            out_node = self.nodes[out_node]
            in_node = self.nodes[in_node]
            if not self.check_cycle(in_node, {out_node}, set()):
                out_node.add_successor(in_node)
                new_edge_count += 1
                self.learned_connections.append(connection)
            if new_edge_count >= num_new_edges:
                break
        # Remove edges
        remove_edge_count = 0
        for _ in range(num_remove_edges * 10):
            if len(self.learned_connections) == 0:
                break
            connection = random.choice(self.learned_connections)
            out_node, in_node = connection
            out_node = self.nodes[out_node]
            in_node = self.nodes[in_node]
            if in_node in self.output_nodes and len(in_node.predecessors) == 1:
                continue
            out_node.remove_successor(in_node)
            remove_edge_count += 1
            self.learned_connections.remove(connection)
            if remove_edge_count >= num_remove_edges:
                break

    def check_cycle(self, new_node, target_nodes, visited=None, rec_stack=None):
        if new_node in target_nodes:
            return True
        for successor in new_node.successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False
