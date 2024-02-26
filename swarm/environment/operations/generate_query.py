#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from copy import deepcopy
from pytube import YouTube
from collections import defaultdict
from typing import List, Any, Optional

from swarm.llm.format import Message
from swarm.graph import Node
from swarm.memory.memory import GlobalMemory
from swarm.utils.log import logger, swarmlog
from swarm.utils.globals import Cost
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.llm import LLMRegistry


class GenerateQuery(Node):
    def __init__(self, 
                 domain: str,
                 model_name: Optional[str] = None,
                 operation_description: str = "Given a question, return what infomation is needed to answer the question.",
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

    def extract_urls(self, text):
        # Regular expression for matching URLs
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, text)
        return urls
    
    def is_youtube_url(self, url: str) -> bool:
        youtube_regex = (
            r'(https?://)?(www\.)?'
            '(youtube|youtu|youtube-nocookie)\.(com|be)/'
            '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
        )
        return bool(re.match(youtube_regex, url))

    def _youtube_download(self, url: str) -> str:
        try:
            video_id = url.split('v=')[-1].split('&')[0]
            video_id = video_id.strip()
            youtube = YouTube(url)
            video_stream = youtube.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            if not video_stream:
                raise ValueError("No suitable video stream found.")
            
            output_dir = "workspace/tmp"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/{video_id}.mp4"
            video_stream.download(output_path=output_dir, filename=f"{video_id}.mp4")
            return output_path
        
        except Exception as e:
            logger.error(f"Error downloading video from {url}: {e}")  # Use logger for error messages
            return ""

    def meta_prompt(self, input, meta_init=False):

        self.prompt_set = PromptSetRegistry.get(self.domain)
        role = self.prompt_set.get_role()
        constraint = self.prompt_set.get_constraint()

        # question = self.prompt_set.get_combine_materials(self.materials)
        prompt = self.prompt_set.get_query_prompt(question=input["task"])    

        if meta_init:
            pass #TODO

        return role, constraint, prompt

    async def _execute(self, inputs: List[Any] = [], **kwargs):

        node_inputs = self.process_input(inputs)
        outputs = []

        for input in node_inputs:
            urls = self.extract_urls(input["task"])

            download_paths = []

            # Process each URL
            for url in urls:
                if self.is_youtube_url(url):
                    download_path = self._youtube_download(url)
                    if download_path:
                        download_paths.append(download_path)

            if urls != []:
                logger.info(urls)
            if download_paths != []:
                logger.info(download_paths)


            files = input.get("files", [])
            if not isinstance(files, list):
                files = []
            files.extend(download_paths)

            role, constraint, prompt = self.meta_prompt(input)

            message = [Message(role="system", content=f"You are a {role}."),
                       Message(role="user", content=prompt)]
            
            response = await self.llm.agen(message)

            executions =  {"operation": self.node_name,
                           "task": input["task"], 
                           "files": files,
                           "input": input.get("task", None), 
                           "subtask": prompt,
                           "output": response,
                           "format": "natural language"}
            outputs.append(executions)
            self.memory.add(self.id, executions)
            
        self.log()
        return outputs
