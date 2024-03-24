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


"""
Imagine someone who has to answer questions.
They can be any person.
Make a list of their possible specializations or social roles.
Make the list as diverse as possible so that you expect them to answer the same question differently.
Make a list of 20, list items only, no need for a description.
"""

class SpecialistAnswer(Node): 
    role_list = [
        "Botanist",
        "Data Scientist",
        "Social Worker",
        "Journalist",
        "Pilot",
        "Anthropologist",
        "Fitness Coach",
        "Politician",
        "Artist",
        "Marine Biologist",
        "Ethicist",
        "Entrepreneur",
        "Linguist",
        "Archaeologist",
        "Nurse",
        "Graphic Designer",
        "Philanthropist",
        "Meteorologist",
        "Sommelier",
        "Cybersecurity Expert"
    ]

    def __init__(self, 
                 domain: str,
                 model_name: Optional[str],
                 operation_description: str = "Aswer as if you were a specialist in <something>.",
                 max_token: int = 50, 
                 id=None):
        super().__init__(operation_description, id, True)
        self.domain = domain
        self.model_name = model_name
        self.llm = LLMRegistry.get(model_name)
        self.max_token = max_token
        self.prompt_set = PromptSetRegistry.get(domain)

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

        for input in node_inputs:
            task = input["task"]
            _, constraint = await self.node_optimize(input, meta_optmize=False)

            # Override role with a specialist role.
            idx_role = hash(self.id) % len(self.role_list)
            role = self.role_list[idx_role]
            print(role)

            system_message = f"You are a {role}. {constraint}. Aswer with one of the 4 letters only: A, B, C or D."

            prompt = self.prompt_set.get_answer_prompt(question=task)
            message = [Message(role="system", content=system_message),
                       Message(role="user", content=prompt)]
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