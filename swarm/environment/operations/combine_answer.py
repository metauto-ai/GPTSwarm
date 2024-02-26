#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import defaultdict
from typing import List, Any, Optional

from swarm.llm.format import Message
from swarm.graph import Node
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.llm import LLMRegistry
from swarm.utils.log import logger, swarmlog
from swarm.utils.globals import Cost


class CombineAnswer(Node):
    def __init__(self, 
                 domain: str,
                 model_name: Optional[str] = None,
                 operation_description: str = "Combine multiple inputs into one.", 
                 max_token: int = 500,
                 id=None):
        super().__init__(operation_description, id, True)
        self.domain = domain
        self.llm = LLMRegistry.get(model_name)
        self.max_token = max_token
        self.prompt_set = PromptSetRegistry.get(self.domain)
        self.role = self.prompt_set.get_role()
        self.constraint = self.prompt_set.get_constraint()


    @property
    def node_name(self):
        return self.__class__.__name__

    def meta_prompt(self, node_inputs, meta_init=False):

        self.materials = defaultdict(str)
        for input in node_inputs:
            operation = input.get('operation')
            if operation:
                self.materials[operation] += f'{input.get("output", "")}\n'
            # if operation == "FileAnalyse":
            #     files_list = input.get("files", [])
            #     self.materials["files"] = "\n".join(files_list)

            self.materials["task"] = input.get('task') 

        question = self.prompt_set.get_combine_materials(self.materials)
        prompt = self.prompt_set.get_answer_prompt(question=question)    

        if meta_init:
            # According to node_inputs and memory history,
            # rewrite the meta_role, meta_constraint and meta_prompt
            pass

        return self.role, self.constraint, prompt


    async def _execute(self, inputs: List[Any] = [], **kwargs):

        node_inputs = self.process_input(inputs)

        role, constraint, prompt = self.meta_prompt(node_inputs, meta_init=False)
        
        message = [Message(role="system", content=f"You are a {role}. {constraint}"),
                Message(role="user", content=prompt)]
        response = await self.llm.agen(message, max_tokens=self.max_token)
    
        executions = {"operation": self.node_name,
            "task": self.materials["task"], 
            "files": self.materials["files"],
            "input": node_inputs, 
            "subtask": prompt,
            "output": response,
            "format": "natural language"}

        self.memory.add(self.id, executions)

        self.log()
        return [executions]
        #return executions
    
    