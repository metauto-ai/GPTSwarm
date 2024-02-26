#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Any, Optional
import asyncio

from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.llm import LLMRegistry
from swarm.environment.operations.crosswords.crosswords_operation import CrosswordsOperation
from swarm.environment.operations.operation_registry import OperationRegistry


@OperationRegistry.register("ReturnAll")
class ReturnAll(CrosswordsOperation):
    def __init__(self, 
                 domain: str, 
                 model_name: Optional[str] = None,
                 operation_description: str = "Take the best solution.",
                 id=None,
                 best_state=True):
        
        super().__init__(operation_description, id, True)
        self.domain = domain
        self.llm = LLMRegistry.get(model_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.best_state = best_state

    async def _execute(self, inputs: List[Any] = [], **kwargs):
        return inputs