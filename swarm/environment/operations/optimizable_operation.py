# The prompts in this script are modified from https://arxiv.org/pdf/2305.03495.pdf#page=10&zoom=100,88,968
import random
import asyncio
from tqdm import tqdm
from typing import Optional

from swarm.graph import Node
from swarm.llm.format import Message
from swarm.llm import LLMRegistry


class OptimizableOperation(Node):
    def __init__(self, 
                 domain: str,
                 combine_inputs_as_one: bool,
                 prompt: str,
                 model_name: Optional[str] = None,
                 operation_description: str = "",
                 id=None,
                 max_domenstrations: int = 4,
                 ):
        self.domain = domain
        self.model_name = model_name
        self.llm = LLMRegistry.get(model_name)
        super().__init__(operation_description, id, combine_inputs_as_one)
        self.operation_description = operation_description
        self.prompt = prompt
        self.domenstrations = []
        self.max_domenstrations = max_domenstrations

    def get_complete_prompt(self, inputs):
        pass

    async def evaluate(self, candidate) -> float:
        raise NotImplementedError

    async def get_new_prompt(self, negative_examples):
        tasks = []
        for negative_example in negative_examples:
            meta_prompt = f""" Here is an example when {self.operation_description} gets wrong.
Input:
{negative_example['input']}
------------------
The output was:
{negative_example['output']}
------------------
It received the following feedback:
{negative_example['feedback']}
"""
            tasks.append(self.llm.agen([Message(role="user", content=meta_prompt), 
                                        Message(role='user', content=f"Identify a problem in {self.operation_description} from the given example and suggest how to prevent it without mentioning the specific example. Respond only one sentence.")], 
                                        max_tokens=100))
    
        responds = await asyncio.gather(*tasks)
        advice = ''
        for i, respond in enumerate(responds):
            advice += f"{i + 1}. {respond}\n"

        meta_prompt = f"""I'm trying to define {self.operation_description} by prompting.
My current prompt is:
"{self.prompt}"

To generate an improved prompt, consider the following:
{advice}
Generate an improved prompt within five sentences. Do not mention a specific task in the prompt!
The prompt should be wrapped with <START> and <END>.
"""
        new_prompt = await self.llm.agen([Message(role="user", content=meta_prompt)], max_tokens=200)
        new_prompt = new_prompt.split("<END>")[0].split("<START>")[-1].strip()
        return new_prompt
    