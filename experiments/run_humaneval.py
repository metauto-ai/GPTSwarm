#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import json
import time
import asyncio
from pathlib import Path

from swarm.environment.tools.reader.readers import JSONLReader, YAMLReader
from swarm.environment.agents.humaneval.code_react import CodeReact
from swarm.environment.tools.coding.python_executor import PyExecutor
from swarm.memory.memory import GlobalMemory
from swarm.utils.globals import Time
from swarm.utils.const import GPTSWARM_ROOT
from swarm.utils.log import logger
from swarm.environment.operations.optimizable_operation import OptimizableOperation
from swarm.optimizer.node_optimizer.node_optimization import optimize


def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w') as file:
            json.dump([], file)

    with open(result_file, 'r') as file:
        data = json.load(file)
    return data

def dataloader(data_list):
    for data in data_list:
        yield data

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def parse_args():
    parser = argparse.ArgumentParser(description="GPTSwarm Experiments on HumanEval")
    parser.add_argument("--config", type=str, help="Path to configuration YAML file.")
    parser.add_argument("--dataset_json", type=str, default="datasets/humaneval/humaneval-py.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--learn_prompt", type=bool, default=False)
    parser.add_argument("--learn_demonstration", type=bool, default=False)

    args = parser.parse_args()
    result_path = GPTSWARM_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if args.config:
        config_args = YAMLReader.parse(args.config, return_str=False)
        for key, value in config_args.items():
            setattr(args, key, value)
    return args

async def main():
    args = parse_args()
    result_file = None

    dataset = JSONLReader.parse_file(args.dataset_json)

    ####################################


    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{GPTSWARM_ROOT}/result/eval")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{'' if args.learn_prompt else 'not'}_learn_prompt_{'' if args.learn_demonstration else 'not'}_learn_demo_{args.llm}_{current_time}.json"
    agent = CodeReact(domain="humaneval", 
                   model_name=args.llm,
                   )
    memory = GlobalMemory.instance()
    ####################################
    opt_frequency = 4
    for i, item in enumerate(dataloader(dataset)):
        task = item["prompt"]
        tests = item["test"]
        inputs = {"task": task, "tests": tests}

        # Agent
        answer = await agent.run(inputs=inputs)
        answer = answer[-1]
        # Evaluate the answer against the test cases here

        data = load_result(result_file)
        total_solved, total_executed = (0, 0) if not data else (data[-1]["Total solved"], data[-1]["Total executed"])

        is_solved, _, _ = PyExecutor().execute(answer, [tests], timeout=100)
        memory.add(task, is_solved)

        total_solved = total_solved + is_solved
        total_executed = total_executed + 1
        accuracy = total_solved/ total_executed

        logger.info(f"total_solved: \n{total_solved}")
        logger.info(f"total_executed: \n{total_executed}")
        logger.info(f"accuracy: \n{accuracy}")

        updated_item = {
            "Question": task,
            "Tests": tests,
            "Attempt answer": answer,
            "Solved": is_solved,
            "Solution": answer,
            "Total solved": total_solved,
            "Total executed": total_executed,
            "Accuracy": accuracy
        }
        data.append(updated_item)

        with open(result_file, 'w') as file:
            json.dump(data, file, indent=4)

        if i % opt_frequency == opt_frequency - 1 and (args.learn_prompt or args.learn_demonstration):
            tasks = [optimize(node, args.learn_demonstration, args.learn_prompt) for node in agent.nodes.values() if isinstance(node, OptimizableOperation)]
            await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())
