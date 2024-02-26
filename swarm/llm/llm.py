from abc import ABC, abstractmethod
from typing import List, Union, Optional

from swarm.llm.format import Message


class LLM(ABC):
    DEFAULT_MAX_TOKENS = 1000
    DEFAULT_TEMPERATURE = 0.2
    DEFUALT_NUM_COMPLETIONS = 1

    @abstractmethod
    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        pass

    @abstractmethod
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        pass
