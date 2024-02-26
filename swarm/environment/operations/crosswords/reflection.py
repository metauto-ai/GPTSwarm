#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import List, Any, Dict, Optional

from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.environment.operations.crosswords.crosswords_operation import CrosswordsOperation
from swarm.environment.domain.crosswords.env import MiniCrosswordsEnv
from swarm.llm import LLMRegistry


class Reflection(CrosswordsOperation):
    def __init__(self, 
                 domain: str, 
                 model_name: Optional[str] = None,
                 operation_description: str = "Learn from a solution.",
                 id=None,
                 branch_factor=3,
                 prune=True):
        
        super().__init__(operation_description, id, False)
        self.domain = domain
        self.llm = LLMRegistry.get(model_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.branch_factor = branch_factor
        self.prune = prune

    async def _execute(self, inputs: List[Any] = [], **kwargs) -> List[Dict[str, MiniCrosswordsEnv]]:
        llm_querier = self.llm_query_with_cache
        env = deepcopy(inputs["env"])
        await env.evaluate(llm_querier, self.prompt_set.get_if_correct_prompt, self.prompt_set.get_value_prompt)
        impossible_words = env.impossible_words
        correct_words = env.correct_words
        incorrect_words = env.incorrect_words
        if len(impossible_words) + len(correct_words) + len(incorrect_words) == 0:
            env.reset()
            return [{'env': env}]
        impossible_words_str = '\n'.join([f'{idx}{word} -- {meaning}' for idx, word, meaning in impossible_words])
        correct_words_str = '\n'.join([f'{idx}{word} -- {meaning}' for idx, word, meaning in correct_words])
        incorrect_words_str = '\n'.join([f'{idx}{word} -- {meaning}' for idx, word, meaning in incorrect_words])
        prompt = self.prompt_set.get_suggest_prompt(env.render_board(), 
                                                            impossible_words_str, 
                                                            correct_words_str, 
                                                            incorrect_words_str)
        
        response = await llm_querier(prompt)
        env.reset()
        env.hints.append(response)
        return [{'env': env}]