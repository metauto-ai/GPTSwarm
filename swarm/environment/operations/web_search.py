#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from copy import deepcopy
from collections import defaultdict
from dotenv import load_dotenv
from typing import List, Any, Optional

from swarm.llm.format import Message
from swarm.graph import Node
from swarm.environment import GoogleSearchEngine, SearchAPIEngine, BingSearchEngine
from swarm.utils.log import logger, swarmlog
from swarm.utils.globals import Cost
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.llm import LLMRegistry

class WebSearch(Node):
    def __init__(self, 
                 domain: str,
                 model_name: Optional[str] = None,
                 operation_description: str = "Given a question, search the web for infomation.",
                 id=None):
        super().__init__(operation_description, id, True)
        self.domain = domain
        self.llm = LLMRegistry.get(model_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role()
        self.constraint = self.prompt_set.get_constraint()
        self.searcher =self._get_searcher()

    @property
    def node_name(self):
        return self.__class__.__name__
    
    def _get_searcher(self):
        load_dotenv()
        if os.getenv("BING_API_KEY"):
            return BingSearchEngine()
        if os.getenv("SEARCHAPI_API_KEY"):
            return SearchAPIEngine()
        if os.getenv("GOOGLE_API_KEY"):
            return GoogleSearchEngine()

    async def _execute(self, inputs: List[Any] = [], max_keywords: int = 5, **kwargs):

        node_inputs = self.process_input(inputs)
        outputs = []
        for input in node_inputs:

            task = input["task"]
            query = input['output']
            prompt = self.prompt_set.get_websearch_prompt(question=task, query=query)

            message = [Message(role="system", content=f"You are a {self.role}."),
                       Message(role="user", content=prompt)]
            generated_quires = await self.llm.agen(message)


            generated_quires = generated_quires.split(',')[:max_keywords]
            logger.info(f"The search keywords include: {generated_quires}")
            search_results = [self.web_search(query) for query in generated_quires]


            logger.info(f"The search results: {search_results}")



            distill_prompt = self.prompt_set.get_distill_websearch_prompt(
               question=input["task"], query=query, results='.\n'.join(search_results))
            
            response = await self.llm.agen(distill_prompt)

            executions =  {"operation": self.node_name,
                            "task": task, 
                            "files": input.get("files", []),
                            "input": query, 
                            "subtask": distill_prompt,
                            "output": response,
                            "format": "natural language"}

            self.memory.add(self.id, executions)
            outputs.append(executions)

        self.log()
        return outputs

    def web_search(self, query):
        return self.searcher.search(query)
    

