# !/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

from swarm.graph import Graph
from swarm.environment.operations import CodeWriting
from swarm.environment.agents.agent_registry import AgentRegistry

@AgentRegistry.register('CodeReact')
class CodeReact(Graph):
    def __init__(self, domain: str, model_name: Optional[str] = None, meta_prompt: bool = False, num_reacts: int = 1):
        self.num_reacts = num_reacts
        super().__init__(domain, model_name, meta_prompt)
    def build_graph(self):

        code_writing = CodeWriting(self.domain, self.model_name)
        self.add_node(code_writing)
        last_node = code_writing
        for _ in range(self.num_reacts):
            code_rewrite = CodeWriting(self.domain, self.model_name)
            last_node.add_successor(code_rewrite)
            last_node = code_rewrite
            self.add_node(code_rewrite)

        self.input_nodes = [code_writing]
        self.output_nodes = [code_rewrite]
