#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import openai
import jsonlines
import re
import regex
import func_timeout
from typing import Union
import random

from typing import List
from swarm.utils.const import GPTSWARM_ROOT

def load_agents_info(candidates_path: str, agent_num: int):
    """
    Load agents' information from a given file path.

    :param society_path: Path to the society file.
    :param agent_num: Number of agents to be loaded.
    :return: Tuple of names and profiles.
    """
    print(candidates_path)
    with open(candidates_path, "r", encoding="utf-8") as file:
        data = json.load(file)["agents"]
        names = [agent['name'] for agent in data[:agent_num]]
        profiles = [agent['profile'] for agent in data[:agent_num]]
        strategies = [agent['strategy'] for agent in data[:agent_num]]

        return names, profiles, strategies
