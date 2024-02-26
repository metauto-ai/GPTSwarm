#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

from swarm.graph import Graph
from swarm.environment.operations.crosswords import BranchingStep, ReturnAll
from swarm.environment.agents.agent_registry import AgentRegistry


@AgentRegistry.register('CrosswordsToT')
class CrosswordsToT(Graph):
    def __init__(self, domain: str, model_name: Optional[str] = None, meta_prompt: bool = False, depth=8, branch_factor=2, prune=True):
        self.depth = depth
        self.branch_factor = branch_factor
        self.prune = prune
        super().__init__(domain, model_name, meta_prompt)

    def build_graph(self):
        step = BranchingStep(self.domain, self.model_name, branch_factor=self.branch_factor, prune=self.prune)
        self.add_node(step)
        self.input_nodes = [step]
        for _ in range(self.depth - 1):
            next_step = BranchingStep(self.domain, self.model_name, branch_factor=self.branch_factor, prune=self.prune)
            self.add_node(next_step)
            step.add_successor(next_step)
            step = next_step
        
        take_best = ReturnAll(self.domain, self.model_name)
        self.add_node(take_best)
        step.add_successor(take_best)
        self.output_nodes = [take_best]