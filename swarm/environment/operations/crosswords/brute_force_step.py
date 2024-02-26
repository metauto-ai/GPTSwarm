#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import List, Any, Optional

from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.environment.domain.crosswords.parser import parse_response
from swarm.llm import LLMRegistry
from swarm.environment.operations.crosswords.crosswords_operation import CrosswordsOperation


class BruteForceStep(CrosswordsOperation):
    def __init__(self, 
                 domain: str, 
                 model_name: Optional[str] = None,
                 operation_description: str = "Perform a brute force step.",
                 id=None,
                 max_candidates=30):
        
        self.domain = domain
        self.llm = LLMRegistry.get(model_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.max_candidates = max_candidates
        super().__init__(operation_description, id, False)

    def brute_force_optimize(self, candidates, scores, env):
        if len(candidates) == 0:
            return 0, env
        candidate = candidates[0]
        candidate_score = scores[0]
        best_score, best_env = self.brute_force_optimize(candidates[1:], scores[1:], deepcopy(env))
        try:
            env.step(candidate, allow_change=False)
            later_score, later_env = self.brute_force_optimize(candidates[1:], scores[1:], deepcopy(env))
            later_score += candidate_score
            if later_score > best_score:
                return later_score, later_env
            else:
                return best_score, best_env
        except:
            return best_score, best_env
        
    async def _execute(self, inputs: List[Any] = [], **kwargs):
        llm_querier = self.llm_query_with_cache
        env = deepcopy(inputs["env"])
        prompt = self.prompt_set.get_propose_prompt(env.render())
        response = await llm_querier(prompt)
        candidates = parse_response(response)[:self.max_candidates]
        _, env = self.brute_force_optimize([candidate for candidate, _ in candidates], [score for _, score in candidates], env)

        return [{'env': env}]
