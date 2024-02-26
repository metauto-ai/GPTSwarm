#!/usr/bin/env python
# -*- coding: utf-8 -*-

from swarm.graph import Graph
from swarm.environment.operations.cot_step import CoTStep
from swarm.environment.agents.agent_registry import AgentRegistry


@AgentRegistry.register('COT')
class COT(Graph):

    def build_graph(self):

        num_thoughts = 3

        assert num_thoughts >= 2

        thoughts = []
        for i_thought in range(num_thoughts):
            thought = CoTStep(self.domain,
                           self.model_name,
                           is_last_step=i_thought==num_thoughts-1)
            if i_thought > 0:
                thoughts[-1].add_successor(thought)
            thoughts.append(thought)

        self.input_nodes = [thoughts[0]]
        self.output_nodes = [thoughts[-1]]

        for thought in thoughts:
            self.add_node(thought)
