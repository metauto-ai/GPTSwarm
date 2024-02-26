#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
from swarm.utils.log import logger

# Global constant should be capitalized and described.
PLAN_TEMPLATE = """
User needs you to give a plan to a specific task.

Related Filenames:
{filenames}

User's Task:
{task}

Available Operations and Their Descriptions in the Environment:
{environment}

Please help outline a plan and identify the necessary operations for each step.

Follow the format provided in these SAMPLES:

START
====
Task: Provide a comprehensive introduction to GPTSwarm and identify if any websites are discussing it.
Files: ["./datasets/demos/gptswarm.txt"]
----
Plan: Begin by using the Reader to load "./datasets/demos/gptswarm.txt" for detailed information, followed by a Websearch to find information using the keyword "GPTSwarm."
----
Operations, Purposes and Inputs:
1. Reader ### To load and analyze the file for relevant information ### Files: ["./datasets/demos/gptswarm.txt"]
2. Search ### To find information related to "GPTSwarm" ### Keywords: ["GPTSwarm Introduction"]
3. Thought ### To complete the task using gathered information ### Inputs: ["DYNAMIC_CHANGE"]
====
END

Please organize your plan accordingly.
"""


def operation_parser(operations_list):

    if not isinstance(operations_list, list):
        raise ValueError("The operations_list must be a list.")

    tools = []
    targets = []
    formatted_inputs = []

    for item in operations_list:
        if not isinstance(item, str):
            continue

        parts = item.split("###")
        if len(parts) != 3:
            continue

        tool, target, input_str = [part.strip() for part in parts]
        tools.append(tool)
        targets.append(target)

        try:
            # Attempt to parse the string as a dictionary
            parsed_input = ast.literal_eval(input_str)
            if isinstance(parsed_input, dict):
                input_dict =  parsed_input
        except (SyntaxError, ValueError):
            # If parsing fails, treat as a simple key-value pair
            if ":" in input_str:
                key, value = input_str.split(":", 1)
                values = value.strip("[] ").replace('"', '').split(", ")
                input_dict =  {key.strip(): values if len(values) > 1 else values[0]}
            else:
                input_dict = {'Inputs': input_str}

        formatted_inputs.append(input_dict)

    return tools, targets, formatted_inputs


def plan_parser(plan_input):

    if not isinstance(plan_input, str):
        raise ValueError("Input must be a string.")
    if not plan_input.strip():
        raise ValueError("Input string is empty.")

    lines = plan_input.split('\n')
    plan = ''
    operations = []
    current_section = None

    for line in lines:
        if 'Plan:' in line:
            current_section = 'plan'
        elif 'Operations, Purposes and Inputs:' in line:
            current_section = 'operations'
            continue

        if current_section == 'plan' and line.strip() != '----':
            plan += line.strip() + ' '
        elif current_section == 'operations' and line.strip() != '====' and not line.strip().startswith('END'):
            operations.append(line.strip())

    plan = plan.strip("Plan:").strip()
    if not plan:
        raise ValueError("Plan section is empty.")
    if not operations:
        raise ValueError("Operations, Purposes and Inputs section is empty.")

    logger.info(operations)

    tools, targets, inputs = operation_parser(operations)

    return {'Plan': plan, 'Operations': tools, 'Targets': targets, 'Inputs': inputs}
