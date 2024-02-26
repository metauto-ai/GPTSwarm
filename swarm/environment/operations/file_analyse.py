#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import List, Any, Optional
from collections import defaultdict

from swarm.llm.format import Message
from swarm.graph import Node
from swarm.memory.memory import GlobalMemory
from swarm.environment import GeneralReader
from swarm.environment.tools.reader.readers import IMGReader, VideoReader
from swarm.utils.log import logger, swarmlog
from swarm.utils.globals import Cost
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.llm import LLMRegistry


reader = GeneralReader()

class FileAnalyse(Node):
    def __init__(self, 
                 domain: str,
                 model_name: Optional[str] = None,
                 operation_description: str = "Given a question, extract infomation from a file.",
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

    async def _execute(self, inputs: List[Any] = [], **kwargs):

        node_inputs = self.process_input(inputs)
        outputs = []
        for input in node_inputs:
            query = input.get("output", "Please organize the information of this file.")
            file = input["files"]
            response = await self.file_analyse(query, file, self.llm)

            executions = {
                "operation": self.node_name,
                "task": input["task"], 
                "files": file,
                "input": query, 
                "subtask": f"Read the content of ###{file}, use query ###{query}",
                "output": response,
                "format": "natural language"
            }

            outputs.append(executions)
            self.memory.add(self.id, executions)

        self.log()
        return outputs


    async def file_analyse(self, query, files, llm):
        answer = ''
        for file in files:
            response = reader.read(query, file)
            if not (isinstance(reader.file_reader, IMGReader) or isinstance(reader.file_reader, VideoReader)):
                prompt = self.prompt_set.get_file_analysis_prompt(query=query, file=response)
                response = await llm.agen(prompt)
            answer += response + '\n'
        return answer
