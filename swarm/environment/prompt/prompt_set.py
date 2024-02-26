#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Any
from abc import ABC, abstractmethod


class PromptSet(ABC):
    """
    Abstract base class for a set of prompts.
    """
    @staticmethod
    @abstractmethod
    def get_role() -> str:
        """ TODO """

    @staticmethod
    @abstractmethod
    def get_constraint() -> str:
        """ TODO """

    @staticmethod
    @abstractmethod
    def get_format() -> str:
        """ TODO """

    @staticmethod
    @abstractmethod
    def get_answer_prompt(question) -> str:
        """ TODO """

    @staticmethod
    @abstractmethod
    def get_adversarial_answer_prompt(question) -> str:
        """ TODO """

    @staticmethod
    @abstractmethod
    def get_query_prompt(question) -> str:
        """ TODO """

    @staticmethod
    @abstractmethod
    def get_file_analysis_prompt(query, file) -> str:
        """ TODO """

    @staticmethod
    @abstractmethod
    def get_websearch_prompt(query) -> str:
        """ TODO """

    @staticmethod
    @abstractmethod
    def get_distill_websearch_prompt(query, results) -> str:
        """ TODO """

    @staticmethod
    @abstractmethod
    def get_reflect_prompt(question, answer) -> str:
        """ TODO """

    @staticmethod
    def get_react_prompt(question, solutions, feedback) -> str:
        """ TODO """

    # @staticmethod
    # @abstractmethod
    # def get_self_consistency(materials: Dict[str, Any]) -> str:
    #     """ TODO """

    # @staticmethod
    # @abstractmethod
    # def get_select_best(materials: Dict[str, Any]) -> str:
    #     """ TODO """

    @staticmethod
    @abstractmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        """ TODO """


