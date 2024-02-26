#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import NamedTuple, List, Tuple
from abc import ABC, abstractmethod

class ExecuteResult(NamedTuple):
    is_passing: bool
    feedback: str
    state: Tuple[bool]

class Executor(ABC):
    @abstractmethod
    def execute(self, func: str, tests: List[str], timeout: int = 5) -> ExecuteResult:
        ...

    @abstractmethod
    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        ...

