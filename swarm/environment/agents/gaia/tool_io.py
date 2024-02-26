#!/usr/bin/env python
# -*- coding: utf-8 -*-

from swarm.graph import Graph
from swarm.environment.operations import FileAnalyse, GenerateQuery, CombineAnswer
from swarm.environment.agents.agent_registry import AgentRegistry

@AgentRegistry.register('ToolIO')
class ToolIO(Graph):
    def build_graph(self):

        query = GenerateQuery(self.domain, self.model_name)
        file_analysis = FileAnalyse(self.domain, self.model_name)

        query.add_successor(file_analysis)
        combine = CombineAnswer(self.domain, self.model_name, max_token=500)
        file_analysis.add_successor(combine)

        self.input_nodes = [query]
        self.output_nodes = [combine]

        self.add_node(query)
        self.add_node(file_analysis)
        self.add_node(combine)