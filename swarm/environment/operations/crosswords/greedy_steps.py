#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import List, Any, Optional

from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.environment.domain.crosswords.parser import parse_response
from swarm.llm import LLMRegistry
from swarm.environment.operations.crosswords.crosswords_operation import CrosswordsOperation


class GreedySteps(CrosswordsOperation):
    def __init__(self, 
                 domain: str, 
                 model_name: Optional[str] = None,
                 operation_description: str = "Perform greedy steps.",
                 id=None):
        
        super().__init__(operation_description, id, False)
        self.domain = domain
        self.llm = LLMRegistry.get(model_name)
        self.prompt_set = PromptSetRegistry.get(domain)

    async def _execute(self, inputs: List[Any] = [], **kwargs):
        llm_querier = self.llm_query_with_cache
        env = inputs["env"]
        prompt = self.prompt_set.get_propose_prompt(env.render())
        response = await llm_querier(prompt)
        candidates = parse_response(response)
        env = deepcopy(env)
        for candidate, _ in candidates:
            try:
                env.step(candidate, allow_change=False)
            except:
                continue

        return [{'env': env}]
