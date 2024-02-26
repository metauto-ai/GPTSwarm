#!/usr/bin/env python
# -*- coding: utf-8 -*-

from swarm.graph import Graph
from swarm.environment.operations import GenerateQuery, FileAnalyse, WebSearch, CombineAnswer
from swarm.environment.agents.agent_registry import AgentRegistry


@AgentRegistry.register('TOT')
class TOT(Graph):
    def build_graph(self):
        query = GenerateQuery(self.domain, self.model_name)

        file_analysis = FileAnalyse(self.domain, self.model_name)
        web_search = WebSearch(self.domain, self.model_name)

        query.add_successor(file_analysis)
        query.add_successor(web_search)
        
        query_left = GenerateQuery(self.domain, self.model_name)
        file_analysis_left = FileAnalyse(self.domain, self.model_name)
        web_search_left = WebSearch(self.domain, self.model_name)
        
        query_left.add_predecessor(file_analysis)
        query_left.add_successor(file_analysis_left)
        query_left.add_successor(web_search_left)

        query_right = GenerateQuery(self.domain, self.model_name)
        file_analysis_right = FileAnalyse(self.domain, self.model_name)
        web_search_right = WebSearch(self.domain, self.model_name)

        query_right.add_predecessor(web_search)
        query_right.add_successor(file_analysis_right)
        query_right.add_successor(web_search_right)

        combine = CombineAnswer(self.domain, self.model_name)

        file_analysis_left.add_successor(combine)
        web_search_left.add_successor(combine)
        file_analysis_right.add_successor(combine)
        web_search_right.add_successor(combine) 

        self.input_nodes = [query]
        self.output_nodes = [combine]

        self.add_node(query)
        self.add_node(file_analysis)
        self.add_node(web_search)
        self.add_node(query_left)
        self.add_node(query_right)
        self.add_node(file_analysis_left)
        self.add_node(web_search_left)
        self.add_node(file_analysis_right)
        self.add_node(web_search_right)
        self.add_node(combine)