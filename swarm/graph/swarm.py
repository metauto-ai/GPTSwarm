#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Optional, Dict, Any
import asyncio
import shortuuid
import numpy as np
import torch
import copy

from swarm.environment.operations.final_decision import FinalDecision, MergingStrategy
from swarm.optimizer.edge_optimizer.parameterization import EdgeWiseDistribution
from swarm.memory import GlobalMemory
from swarm.graph.composite_graph import CompositeGraph
from swarm.utils.log import logger
from swarm.environment.agents import AgentRegistry
from swarm.environment.operations.operation_registry import OperationRegistry


class Swarm:
    """
    A class representing a swarm in the GPTSwarm framework.

    Attributes:
    """

    def __init__(self, 
                 agent_names: List[str],
                 domain: str, # No default, we want the user to be aware of what domain they select.
                 model_name: Optional[str] = None, # None is mapped to "gpt-4-1106-preview".
                 open_graph_as_html: bool = False,
                 final_node_class: str = "FinalDecision",
                 final_node_kwargs: Dict[str, Any] = {'strategy': MergingStrategy.OutputsAsReferences},
                 edge_optimize: bool = False,
                 node_optimize: bool = False,
                 init_connection_probability: float = 0.5,
                 connect_output_nodes_to_final_node: bool = False,
                 include_inner_agent_connections: bool = True,
                 ):
        
        self.id = shortuuid.ShortUUID().random(length=4)    
        self.agent_names = agent_names
        self.domain = domain
        self.model_name = model_name
        self.open_graph_as_html = open_graph_as_html
        self.memory = GlobalMemory.instance()
        self.final_node_class = final_node_class  
        self.final_node_kwargs = final_node_kwargs
        self.edge_optimize = edge_optimize
        self.node_optimize = node_optimize
        self.init_connection_probability = init_connection_probability
        self.connect_output_nodes_to_final_node = connect_output_nodes_to_final_node
        self.organize(include_inner_agent_connections)

    def organize(self, include_inner_agent_connections: bool = True):

        self.used_agents = []
        decision_method = OperationRegistry.get(self.final_node_class, self.domain, self.model_name, **self.final_node_kwargs)
        self.composite_graph = CompositeGraph(decision_method,
                                              self.domain, self.model_name)
        potential_connections = []

        for agent_name in self.agent_names:
            if agent_name in AgentRegistry.registry:
                agent_instance = AgentRegistry.get(agent_name,
                                                   self.domain, self.model_name)
                if not include_inner_agent_connections:
                    for node in agent_instance.nodes:
                        for successor in agent_instance.nodes[node].successors:
                            potential_connections.append((node, successor.id))
                        agent_instance.nodes[node].successors = []
                self.composite_graph.add_graph(agent_instance)
                self.used_agents.append(agent_instance)
            else:
                logger.error(f"Cannot find {agent_name} in the list of registered agents "
                             f"({list(AgentRegistry.keys())})")
        
        potential_connections = []
        if self.edge_optimize:  
            # Add bi-directional connections between all nodes of all agents (except for the decision nodes).
            for agent1 in self.used_agents:
                for agent2 in self.used_agents:
                    if agent1 != agent2:
                        for node1 in agent1.nodes:
                            for node2 in agent2.nodes:
                                potential_connections.append((node1, node2)) # (from, to)

            # Add only forward connections from all agents' nodes to the final decision node.
            for agent in self.used_agents:
                for node in agent.nodes:
                    if (self.connect_output_nodes_to_final_node and
                            node in [output_node.id for output_node in agent.output_nodes]):
                        agent.nodes[node].add_successor(decision_method)
                    else:
                        potential_connections.append((node, decision_method.id)) # (from, to)
                        
        else:
            # Connect all output nodes to the decision method if edge optimization is not enabled
            for agent in self.used_agents:
                for node in agent.nodes:
                    if node in [output_node.id for output_node in agent.output_nodes]:
                        agent.nodes[node].add_successor(decision_method)

        self.connection_dist = EdgeWiseDistribution(potential_connections, self.init_connection_probability)
        self.potential_connections = potential_connections

    def visualize_adj_matrix_distribution(self, logits):
        probs = torch.sigmoid(logits)
        matrix = np.zeros((self.composite_graph.num_nodes, self.composite_graph.num_nodes))
        num_nodes_per_agent = np.array([len(agent.nodes) for agent in self.used_agents])
        for i in range(len(num_nodes_per_agent)):
            matrix[num_nodes_per_agent[:i].sum():num_nodes_per_agent[:i+1].sum(), num_nodes_per_agent[:i].sum():num_nodes_per_agent[:i+1].sum()] \
                = self.used_agents[i].adj_matrix
        
        probs_idx = 0
        for i in range(len(self.used_agents)):
            for j in range(len(self.used_agents)):
                if i != j:
                    for k in range(num_nodes_per_agent[i]):
                        for l in range(num_nodes_per_agent[j]):
                            matrix[k + num_nodes_per_agent[:i].sum(), l + num_nodes_per_agent[:j].sum()] = probs[probs_idx]
                            probs_idx += 1

        node_idx = 0
        for agent in self.used_agents:
            for node in agent.nodes:
                if node in [output_node.id for output_node in agent.output_nodes] and self.connect_output_nodes_to_final_node:
                    matrix[node_idx, -1] = 1
                else:
                    matrix[node_idx, -1] = probs[probs_idx]
                    probs_idx += 1
                node_idx += 1

        return matrix

    def run(self,
            inputs: Dict[str, Any],
            realized_graph: Optional[CompositeGraph] = None,
            display: bool = False,
            ):

        if realized_graph is None:
            _graph, _ = self.connection_dist.realize(self.composite_graph)
        else:
            _graph = copy.deepcopy(realized_graph)

        if display:
            _graph.display(draw=self.open_graph_as_html)

        final_answer = asyncio.run(_graph.run(inputs))

        return final_answer

    async def arun(self,
             inputs: Dict[str, Any],
             realized_graph: Optional[CompositeGraph] = None,
             ):

        if realized_graph is None:
            _graph, _ = self.connection_dist.realize(self.composite_graph)
        else:
            _graph = copy.deepcopy(realized_graph)

        _graph.display(draw=self.open_graph_as_html)

        final_answer = await _graph.run(inputs)

        return final_answer
