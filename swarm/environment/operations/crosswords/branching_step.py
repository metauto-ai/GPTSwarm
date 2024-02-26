#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
import asyncio
from typing import List, Any, Optional

from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.environment.domain.crosswords.parser import parse_response
from swarm.llm import LLMRegistry
from swarm.environment.operations.crosswords.crosswords_operation import CrosswordsOperation


class BranchingStep(CrosswordsOperation):
    def __init__(self, 
                 domain: str, 
                 model_name: Optional[str] = None,
                 operation_description: str = "Perform a step in tree search.",
                 id=None,
                 branch_factor=3,
                 prune=True):
        
        super().__init__(operation_description, id, False)
        self.domain = domain
        self.llm = LLMRegistry.get(model_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.branch_factor = branch_factor
        self.prune = prune

    async def _execute(self, inputs: List[Any] = [], **kwargs):
        llm_querier = self.llm_query_with_cache
        env = inputs["env"]
        if not env.extendable:
            return [{
                "env": env
            }]
        prompt = self.prompt_set.get_propose_prompt(env.render())
        response = await llm_querier(prompt)
        candidates = parse_response(response)[:self.branch_factor]

        if len(candidates) == 0:
            return [{
                "env": env
            }]

        outputs = []
        next_envs = []
        for candidate, _ in candidates:
            _env = deepcopy(env)
            _env.step(candidate)
            next_envs.append(_env)
        
        if self.prune:
            tasks = []
            for _env in next_envs:
                tasks.append(_env.check_termination(llm_querier, self.prompt_set.get_value_prompt))
            await asyncio.gather(*tasks)

        for _env in next_envs:    
            outputs.append({
                "env": _env
            })
        return outputs
