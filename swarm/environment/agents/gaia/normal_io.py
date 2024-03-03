#!/usr/bin/env python
# -*- coding: utf-8 -*-

from swarm.graph import Graph
from swarm.environment.operations import DirectAnswer
from swarm.environment.agents.agent_registry import AgentRegistry


@AgentRegistry.register('NormalIO')
class NormalIO(Graph):
    def build_graph(self):
        io = DirectAnswer(self.domain, self.model_name, max_token=500)
        self.add_node(io)
        self.input_nodes = [io]
        self.output_nodes = [io]
