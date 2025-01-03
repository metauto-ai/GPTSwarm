#!/usr/bin/env python
# -*- coding: utf-8 -*-

from swarm.graph import Graph
from swarm.environment.operations import SpecialistAnswer
from swarm.environment.agents.agent_registry import AgentRegistry


@AgentRegistry.register('SpecialistAgent')
class SpecialistAgent(Graph):
    def build_graph(self):
        sa = SpecialistAnswer(self.domain, self.model_name)
        self.add_node(sa)
        self.input_nodes = [sa]
        self.output_nodes = [sa]
