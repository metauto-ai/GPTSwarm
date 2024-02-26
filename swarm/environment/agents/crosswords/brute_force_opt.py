#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

from swarm.graph import Graph
from swarm.environment.operations.crosswords import BruteForceStep
from swarm.environment.agents.agent_registry import AgentRegistry


@AgentRegistry.register('CrosswordsBruteForceOpt')
class CrosswordsBruteForceOpt(Graph):
    def __init__(self, domain: str, model_name: Optional[str] = None, meta_prompt: bool = False, num_iters=3):
        self.num_iters = num_iters
        super().__init__(domain, model_name, meta_prompt)
    def build_graph(self):
        last_step = None
        for _ in range(self.num_iters):
            step = BruteForceStep(self.domain, self.model_name)
            self.add_node(step)
            if last_step:
                last_step.add_successor(step)
                last_step = step
            else:
                self.input_nodes = [step]
                last_step = step
            self.output_nodes = [last_step]