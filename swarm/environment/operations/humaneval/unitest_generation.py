#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import defaultdict
from typing import Any, Union, List, Optional, Callable

from swarm.llm.format import Message
from swarm.graph import Node
from swarm.memory.memory import GlobalMemory
from swarm.utils.log import logger, swarmlog
from swarm.utils.globals import Cost
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.llm import LLMRegistry
from swarm.environment.prompt.human_eval_fewshot import PY_CHECK_EXAMPLES_FEW_SHOT


class UnitestGeneration(Node):
    def __init__(self, 
                 domain: str,
                 model_name: Optional[str] = None,
                 operation_description: str = "Unitest generation.",
                 id=None):
        super().__init__(operation_description, id, True)
        self.domain = domain
        self.llm = LLMRegistry.get(model_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role()
        self.constraint = self.prompt_set.get_constraint()

    @property
    def node_name(self):
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

    def select_examples(self, prompt, combined_examples, model, num_to_select=1):
        """
        Get and parse the selected examples.
        """
        import ast
        raw_selected_examples = self.check_examples(prompt, combined_examples, model, num_to_select)
        return ast.literal_eval(raw_selected_examples)


    async def check_examples(self, prompt, combined_examples: str, max_num_tests: int = 2, 
                       check_examples_few_shot:  Optional[str] = PY_CHECK_EXAMPLES_FEW_SHOT,):

        if check_examples_few_shot is not None:

            user_prompt = f'{check_examples_few_shot}\n\n[problem]:\n```{prompt}```\n\n[candidate examples]:\n{combined_examples}\n\n[selected examples]:'

            messages = [
                Message(
                    role="system",
                    content=(
                        "As an senior programmer with expertise in test case generation, "
                        "You will encounter a coding problem accompanied by potential test cases. "
                        "Identify and select the test cases that are accurate and most pertinent to the presented problem, demonstrating the depth of your understanding."
                    )
                ),
                Message(
                    role="user",
                    content=f'Carefully read the [problem] and review the provided [candidate examples]. Since some cases in [candidate examples] are clearly wrong, select no more than three accurate ones according to the [problem]. \n{check_examples_few_shot}\n\n[problem]:\n```{prompt}```\n\n[candidate examples]:\n{combined_examples}\n\n[selected examples]:',
                )
            ]

        response = await self.llm.agen(messages)

        return response


    def meta_prompt(self, input, meta_init=False):

        self.prompt_set = PromptSetRegistry.get(self.domain)
        role = self.prompt_set.get_role()
        constraint = self.prompt_set.get_constraint()

        subtask = input['subtask']
        answer = input['output']
        prompt = self.prompt_set.get_reflect_prompt(question=subtask, answer=answer)

        if meta_init:
            pass #TODO

        return role, constraint, prompt

    async def _execute(self, inputs: List[Any] = [], **kwargs):


        node_inputs = self.process_input(inputs)
        
        for input in node_inputs:

            role, constraint, prompt = self.meta_prompt(input)
            
            message = [Message(role="system", content=f"You are a {role}. {constraint}"),
                    Message(role="user", content=prompt)]
            
            response = await self.llm.agen(message)
            self.memory.add(self.id, {"operation": self.node_name,
                            #"task_id": input["task_id"], 

                            "task": input["task"], 
                            "files": input.get("files", []),
                            "input": input.get("output", None), 
                            "subtask": prompt,
                            "output": response,
                            "format": "natural language"})
            #self.log()
    