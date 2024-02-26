#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
import asyncio
from collections import defaultdict
import random
from typing import List, Any, Optional

from swarm.llm.format import Message
from swarm.graph import Node
from swarm.memory.memory import GlobalMemory
from swarm.utils.log import logger, swarmlog
from swarm.utils.globals import Cost
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.llm import LLMRegistry
from swarm.optimizer.node_optimizer import MetaPromptOptimizer
from swarm.environment.tools.coding.python_executor import PyExecutor
from swarm.environment.operations.optimizable_operation import OptimizableOperation


class CodeWriting(OptimizableOperation):
    def __init__(self, 
                 domain: str,
                 model_name: Optional[str] = None,
                 operation_description: str = "a Python code generator",
                 id=None):
        prompt = "You are an AI that only responds with only Python code. "
        prompt += "You will be given a function signature and its docstring by the user. "
        prompt += "Write your full implementation (restate the function signature). "
        prompt += "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        super().__init__(domain, False, prompt, model_name, operation_description, id)
        self.domain = domain
        self.model_name = model_name
        self.llm = LLMRegistry.get(model_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role()
        self.constraint = self.prompt_set.get_constraint()

    @property
    def node_name(self):
        """Return the class name."""
        return self.__class__.__name__


    def extract_example(self, prompt: str) -> list:
        lines = (line.strip() for line in prompt.split('\n') if line.strip())

        results = []
        lines_iter = iter(lines)
        for line in lines_iter:
            if line.startswith('>>>'):
                function_call = line[4:]
                expected_output = next(lines_iter, None)
                if expected_output:
                    results.append(f"assert {function_call} == {expected_output}")

        return results

    async def _execute(self, inputs: List[Any] = [], max_tries: int = 1, **kwargs):
        """
        Execute the node with the given inputs.
        """

        node_inputs = self.process_input(inputs)
        node_outputs = []

        for input in node_inputs:
            if input.get('is_solved', False):
                execution = deepcopy(input)
            else:
                task = input["task"]
                if 'feedback' in input.keys():
                    input = self.prompt_set.get_react_prompt(task, input["output"], input["feedback"])
                else:
                    input = input["task"]
                self.internal_tests = self.extract_example(task)
                message = self.get_messages(input, self.prompt, self.domenstrations)
                
                response = await self.llm.agen(message)
                response = response.strip("```python\n").strip("```")
                is_solved, feedback, _ = PyExecutor().execute(response, self.internal_tests, timeout=10)
                execution = {
                    "operation": self.node_name,
                    "task": task, 
                    "input": input,
                    "feedback": feedback,
                    "output": response,
                    "format": "python code",
                    "is_solved": is_solved,
                }

            self.memory.add(self.id, execution)
            node_outputs.append(execution)

        return node_outputs

    def get_messages(self, task, prompt, domenstrations):
        messages = []
        messages.append(Message(role="system", content=prompt))
        for domenstration in domenstrations:
            messages.append(Message(role="user", content=domenstration['input']))
            messages.append(Message(role="assistant", content=domenstration['output']))
        messages.append(Message(role="user", content=task))
        return messages

    async def evaluate(self, candidate):
        prompt, domenstrations = candidate
        inputs = self.memory.query_by_id(self.id)
        inputs = [record for record in self.memory.query_by_id(self.id)[-10:]]#random.sample(inputs, min(10, len(inputs)))#
        score = 0
        tasks = []
        for input in inputs:
            message = self.get_messages(input['input'], prompt, domenstrations)
            response = await self.llm.agen(message)
            response = response.strip("```python\n").strip("```")
            tests = self.extract_example(input['task'])
            is_solved, _, _ = PyExecutor().execute(response, tests, timeout=10)
            score += is_solved

        return score / len(inputs)