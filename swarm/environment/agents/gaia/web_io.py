#!/usr/bin/env python
# -*- coding: utf-8 -*-

from swarm.graph import Graph
from swarm.environment.operations import WebSearch, GenerateQuery, CombineAnswer
from swarm.environment.agents.agent_registry import AgentRegistry

@AgentRegistry.register('WebIO')
class WebIO(Graph):
    def build_graph(self):

        query = GenerateQuery(self.domain, self.model_name)
        websearch = WebSearch(self.domain, self.model_name)

        query.add_successor(websearch)
        combine = CombineAnswer(self.domain, self.model_name, max_token=500)
        websearch.add_successor(combine)

        self.input_nodes = [query]
        self.output_nodes = [combine]

        self.add_node(query)
        self.add_node(websearch)
        self.add_node(combine)