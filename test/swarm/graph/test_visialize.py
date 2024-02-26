import pytest
from dataclasses import dataclass
from typing import Dict

from swarm.graph.visualize import GPTSwarmVis


class Node:
    def __init__(self, operation, successors=[], predecessors=[]):
        self.operation = operation
        self.successors = successors
        self.predecessors = predecessors


@dataclass
class Graph:
    nodes: Dict[str, Node]


def test_visualize():

    test_graph = Graph({
        'A': Node("Start", successors=['B']),
        'B': Node("Process", predecessors=['A'], successors=['C', 'D']),
        'C': Node("Decision", predecessors=['B'], successors=['E']),
        'D': Node("Operation", predecessors=['B'], successors=['E']),
        'E': Node("End", predecessors=['C', 'D'])
    })

    for node_id, node in test_graph.nodes.items():
        node.id = node_id 
        node.successors = [test_graph.nodes[n] for n in node.successors]  
        node.predecessors = [test_graph.nodes[n] for n in node.predecessors]

    GPTSwarmVis(test_graph, dry_run=True)
    

if __name__ == "__main__":
    pytest.main()
