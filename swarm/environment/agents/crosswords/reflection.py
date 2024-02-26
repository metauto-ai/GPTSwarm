#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

from swarm.graph import Graph
from swarm.environment.operations.crosswords import Reflection, GreedySteps
from swarm.environment.agents.agent_registry import AgentRegistry


@AgentRegistry.register('CrosswordsReflection')
class CrosswordsReflection(Graph):
    def __init__(self, domain: str, model_name: Optional[str] = None, meta_prompt: bool = False, num_reflections=1, num_inner_iters=2):
        self.num_reflections = num_reflections
        self.num_inner_iters = num_inner_iters
        super().__init__(domain, model_name, meta_prompt)
    def build_graph(self):
        last_step = None
        for i in range(self.num_reflections + 1):
            for j in range(self.num_inner_iters):
                step = GreedySteps(self.domain, self.model_name)
                self.add_node(step)
                if last_step:
                    last_step.add_successor(step)
                    last_step = step
                else:
                    self.input_nodes = [step]
                    last_step = step
            if i < self.num_reflections:
                reflection = Reflection(self.domain, self.model_name)
                self.add_node(reflection)
                if last_step is None:
                    raise Exception("last_step should not be None")
                last_step.add_successor(reflection)
                last_step = reflection
        if last_step is None:
            raise Exception("last_step should not be None")
        self.output_nodes = [last_step]