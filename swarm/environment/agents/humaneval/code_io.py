#!/usr/bin/env python
# -*- coding: utf-8 -*-

from swarm.graph import Graph
from swarm.environment.operations import CodeWriting
from swarm.environment.agents.agent_registry import AgentRegistry


@AgentRegistry.register('CodeIO')
class CodeIO(Graph):
    def build_graph(self):
        io = CodeWriting(self.domain, self.model_name)
        self.add_node(io)
        self.input_nodes = [io]
        self.output_nodes = [io]
