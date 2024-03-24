#!/usr/bin/env python
# -*- coding: utf-8 -*-

from swarm.graph import Graph
from swarm.environment.operations import SpecialistAnswer
from swarm.environment.agents.agent_registry import AgentRegistry


@AgentRegistry.register('SpecialistAgent')
class SpecialistAgent(Graph):
    def build_graph(self):
        io = SpecialistAnswer(self.domain, self.model_name)
        self.add_node(io)
        self.input_nodes = [io]
        self.output_nodes = [io]
