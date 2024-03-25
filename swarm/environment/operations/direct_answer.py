#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import defaultdict
from swarm.llm.format import Message
from swarm.graph import Node
from swarm.memory.memory import GlobalMemory
from typing import List, Any, Optional
from swarm.utils.log import logger, swarmlog
from swarm.utils.globals import Cost
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.llm.format import Message
from swarm.llm import LLMRegistry
from swarm.optimizer.node_optimizer import MetaPromptOptimizer


class DirectAnswer(Node): 
    def __init__(self, 
                 domain: str,
                 model_name: Optional[str],
                 operation_description: str = "Directly output an answer.",
                 max_token: int = 50, 
                 id=None):
        super().__init__(operation_description, id, True)
        self.domain = domain
        self.model_name = model_name
        self.llm = LLMRegistry.get(model_name)
        self.max_token = max_token
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role()
        self.constraint = self.prompt_set.get_constraint()


    @property
    def node_name(self):
        return self.__class__.__name__
    
    async def node_optimize(self, input, meta_optmize=False):
        task = input["task"]
        self.prompt_set = PromptSetRegistry.get(self.domain)
        role = self.prompt_set.get_role()
        constraint = self.prompt_set.get_constraint()

        if meta_optmize:
            update_role = role 
            node_optmizer = MetaPromptOptimizer(self.model_name, self.node_name)
            update_constraint = await node_optmizer.generate(constraint, task)
            return update_role, update_constraint

        return role, constraint


    async def _execute(self, inputs: List[Any] = [], **kwargs):
        
        node_inputs = self.process_input(inputs)
        outputs = []

        task: Optional[str] = None
        additional_knowledge: List[str] = []
        for input in node_inputs:
            if len(input) == 1 and 'task' in input: # Swarm input
                task = input['task']
            else: # All other incoming edges
                extra_knowledge = f"Opinion of {input['operation']} is \"{input['output']}\"."
                additional_knowledge.append(extra_knowledge)

        if task is None:
            raise ValueError(f"{self.__class__.__name__} expects swarm input among inputs")

        user_message = "\n\n"
        if len(additional_knowledge) > 0:
            for extra_knowledge in additional_knowledge:
                user_message = user_message + extra_knowledge + "\n\n"

        prompt = self.prompt_set.get_answer_prompt(question=task)
        user_message = user_message + prompt

        role, constraint = await self.node_optimize(input, meta_optmize=False)
        system_message = f"You are a {role}. {constraint}"

        message = [Message(role="system", content=system_message),
                    Message(role="user", content=user_message)]
        response = await self.llm.agen(message, max_tokens=self.max_token)

        execution = {
            "operation": self.node_name,
            "task": task,
            "files": input.get("files", []),
            "input": task,
            "role": role,
            "constraint": constraint,
            "prompt": prompt,
            "output": response,
            "ground_truth": input.get("GT", []),
            "format": "natural language"
        }
        outputs.append(execution)
        self.memory.add(self.id, execution)

        # self.log()
        return outputs 