#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import shortuuid
from typing import Any, List, Optional, Dict
from copy import deepcopy
from abc import ABC, abstractmethod
import async_timeout
import numpy as np

from swarm.graph.visualize import GPTSwarmVis
from swarm.memory import GlobalMemory
from swarm.graph.node import Node


class Graph(ABC):
    """
    A framework for managing and executing a network of interconnected nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    Attributes:
        model (LLM): An instance of a language model used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.
        memory (Memory): A memory management system to store and retrieve data related to the graph.
        role (str): The role of the graph, defining its purpose within a larger system.
        constraint (str): Operational constraints that the graph adheres to.
        format (str): The format of responses or data processed by the graph.
        system_content (str): A formatted string that combines role, constraint, and format.
        is_aggregate (bool): Flag indicating whether the graph aggregates data from nodes.
        input_nodes (list): List of nodes designated as input points to the graph.
        output_node (Node): The node designated as the primary output point of the graph.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        display(draw=True): Displays a textual representation of the graph, with an option for a visual representation.
        run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
    """

    def __init__(self, 
                domain: str,
                model_name: Optional[str] = None,
                meta_prompt: bool = False,
                ):

        self.id = shortuuid.ShortUUID().random(length=4)
        self.domain = domain
        self.model_name = model_name
        self.meta_prompt = meta_prompt
        self.nodes = {}
        self.memory = GlobalMemory.instance()
        self.is_aggregate = False
        self.input_nodes: List[Node] = []
        self.output_nodes: List[Node] = []
        self.build_graph()

    @property
    def adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.successors)
        return num_edges
    
    @property
    def num_nodes(self):
        return len(self.nodes)

    @abstractmethod
    def build_graph(self):
        """ To be overriden bu a descendant class """

    def add_node(self, node: Node):
        """
        Creates and adds a new node to the graph.
        If id is not provided, generates a unique id for the node.
        """
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=5)
        node.id = node_id

        self.nodes[node_id] = node
        return node   

    def display(self, draw=True, file_name=None):
        """
        Prints a simple textual representation of the graph.
        """
        # for node in self.nodes.values():
        #     print(f"Node ID: {node.id}, Type: {type(node).__name__}, "
        #           f"Predecessors: {[n.id for n in node.predecessors]}, "
        #           #f"Successors: {[n.id for n in node.successors]}"
        #           )
        if draw:
            GPTSwarmVis(self, file_name=file_name)

    async def run(self, inputs: Dict[str, Any], 
                  max_tries: int = 3, 
                  max_time: int = 600, 
                  return_all_outputs: bool = False) -> List[Any]:
 
        def is_node_useful(node):
            if node in self.output_nodes:
                return True
            
            for successor in node.successors:
                if is_node_useful(successor):
                    return True
            return False
        
        useful_node_ids = [node_id for node_id, node in self.nodes.items() if is_node_useful(node)]
        in_degree = {node_id: len(self.nodes[node_id].predecessors) for node_id in useful_node_ids}
        zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0 and node_id in useful_node_ids]

        for i, input_node in enumerate(self.input_nodes):
            node_input = deepcopy(inputs)
            input_node.inputs = [node_input]

        while zero_in_degree_queue:
            current_node_id = zero_in_degree_queue.pop(0)
            current_node = self.nodes[current_node_id]
            tries = 0
            while tries < max_tries:
                try:
                    await asyncio.wait_for(self.nodes[current_node_id].execute(), timeout=max_time)
                    break
                except asyncio.TimeoutError:
                    print(f"Node {current_node_id} execution timed out, retrying {tries + 1} out of {max_tries}...")
                except Exception as e:
                    print(f"Error during execution of node {current_node_id}: {e}")
                    break
                tries += 1

            for successor in current_node.successors:
                if successor.id in useful_node_ids:
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)

        final_answers = []

        for output_node in self.output_nodes:
            output_messages = output_node.outputs
            if len(output_messages) > 0 and not return_all_outputs:
                final_answer = output_messages[-1].get("output", output_messages[-1])
                final_answers.append(final_answer)
            else:
                for output_message in output_messages:
                    final_answer = output_message.get("output", output_message)
                    final_answers.append(final_answer)

        if len(final_answers) == 0:
            final_answers.append("No answer since there are no inputs provided")
        return final_answers

    def find_node(self, id: str):
        for node in self.nodes.values():
            if node.id == id:
                return node
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")
