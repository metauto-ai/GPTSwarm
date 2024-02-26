import pandas as pd
from typing import Dict, Any, Union, List
from collections.abc import Sequence
from abc import ABC, abstractmethod


SwarmInput = Dict[str, Any]


class BaseDataset(ABC, Sequence[Any]):
    @staticmethod
    @abstractmethod
    def get_domain() -> str:
        """ To be overriden. """

    @abstractmethod
    def split(self) -> str:
        """ To be overriden. """

    @abstractmethod
    def record_to_swarm_input(self, record: pd.DataFrame) -> SwarmInput:
        """ To be overriden. """

    @abstractmethod
    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        """ To be overriden. """

    @abstractmethod
    def record_to_target_answer(self, record: pd.DataFrame) -> str:
        """ To be overriden. """
