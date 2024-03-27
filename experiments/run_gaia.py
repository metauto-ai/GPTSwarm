#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import yaml
import json
import time
import asyncio
from pathlib import Path

from swarm.graph.swarm import Swarm
from swarm.environment.tools.reader.readers import JSONReader, YAMLReader
from swarm.environment.agents.io import IO
from swarm.environment.agents.gaia.normal_io import NormalIO
from swarm.environment.agents.gaia.tool_io import ToolIO
from swarm.environment.agents.gaia.web_io import WebIO
from swarm.environment.agents.gaia.tool_tot import ToolTOT
from swarm.environment.operations import DirectAnswer
from swarm.memory.memory import GlobalMemory
from swarm.utils.globals import Time, Cost, CompletionTokens, PromptTokens
from swarm.utils.const import GPTSWARM_ROOT
from swarm.utils.log import initialize_log_file, logger, swarmlog
from swarm.environment.domain.gaia import question_scorer
from swarm.environment.operations.final_decision import MergingStrategy


def dataloader(data_list):
    for data in data_list:
        yield data

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

async def main():
    parser = argparse.ArgumentParser(description="GPTSwarm Experiments on GAIA")
    parser.add_argument("--config", type=str, help="Path to configuration YAML file.")
    parser.add_argument("--domain", type=str, default="gaia")
    parser.add_argument("--agents", nargs='+', default=["IO"])
    parser.add_argument("--dataset_json", type=str, default="datasets/gaia/level_1_val.json") #level_1_val_solveable.json
    parser.add_argument("--dataset_files", type=str, default="datasets/gaia/val_files")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm", type=str, default="gpt-4-1106-preview") #gpt-4-1106-preview  gpt-3.5-turbo-1106 gpt-3.5-turbo gpt-4
    args = parser.parse_args()

    result_path = GPTSWARM_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time


    log_file_path = initialize_log_file("GAIA", Time.instance().value)

    if args.config:
        config_args = YAMLReader.parse(args.config, return_str=False)
        for key, value in config_args.items():
            setattr(args, key, value)

    start_index = 0
    result_file = None

    dataset = JSONReader.parse_file(args.dataset_json)

    ####################################

    # strategy = MergingStrategy.SelfConsistency #MergingStrategy.SelectBest #MergingStrategy.SelfConsistency #MergingStrategy.SelectBest #MergingStrategy.SelectBest #MergingStrategy.SelfConsistency # MergingStrategy.MajorityVote MergingStrategy.RandomChoice

    experiment_name = "ToolTOT"

    # swarm = Swarm(["ToolTOT"]*7, 
    #               "gaia",
    #               model_name="mock", #args.llm, #"mock", #args.llm,#args.llm,
    #               final_node_class="FinalDecision",
    #               final_node_kwargs=dict(strategy=strategy)
    #             )
    # swarm.composite_graph.display()

    print(args.llm)

    #agent = IO(domain="gaia", model_name=args.llm)
    #agent = WebIO(domain="gaia", model_name=args.llm)
    #agent = ToolIO(domain="gaia", model_name=args.llm)
    agent = ToolTOT(domain="gaia", model_name=args.llm)

    #io = DirectAnswer(domain="gaia", model_name=args.llm)

    agent.display()

    ####################################

    for i, item in enumerate(dataloader(dataset)):
    
        if i < start_index:
            print(f"Skipping index {i}...")
            continue

        start_time = time.time()
        task = item["Question"]
        files = [os.path.join(args.dataset_files, item["file_name"])] if item["file_name"] else item["file_name"]
        ground_truth = item["Final answer"]
        inputs = {"task": task, "files": files, "GT": ground_truth}

        swarmlog("🐝GPTSWARM SYS", f"Finish {i} samples...", Cost.instance().value, PromptTokens.instance().value, CompletionTokens.instance().value, log_file_path)

        # Swarm
        # answer = await swarm.composite_graph.run(inputs)
        # answer = answer[-1].split("FINAL ANSWER: ")[-1]

        # end_time = time.time()
        # exe_time =  end_time - start_time

        # print("-----")
        # print(f"SWARM ANSWER: {answer}")
        # print("-----")

        # Agent
        answer = await agent.run(inputs=inputs)
        answer = answer[-1].split("FINAL ANSWER: ")[-1]

        end_time = time.time()
        exe_time =  end_time - start_time


        print("-----")
        print(f"AGENT ANSWER: {answer}")
        print("-----")
        
        """
        answer = await io._execute(inputs=inputs)
        answer = answer[-1]["output"].split("FINAL ANSWER: ")[-1]

        print("-----")
        print(f"OPERATION ANSWER: {answer}")
        print("-----")
        """

        # current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # Time.instance().value = current_time
        
        result_dir = Path(f"{GPTSWARM_ROOT}/result/eval")
        result_file = result_file or (result_dir / f"{'_'.join(experiment_name.split())}_{args.llm}_{current_time}.json")

        result_dir.mkdir(parents=True, exist_ok=True)

        if not result_file.exists():
            with open(result_file, 'w') as file:
                json.dump([], file)

        with open(result_file, 'r') as file:
            data = json.load(file)

        total_solved, total_executed = (0, 0) if not data else (data[-1]["Total solved"], data[-1]["Total executed"])
        is_solved = question_scorer(answer, item['Final answer'])

        updated_item = {
            "Question": item["Question"],
            "GT": item['Final answer'],
            "Attempt answer": answer,
            "Solved": is_solved,
            "Total solved": total_solved + is_solved,
            "Total executed": total_executed + 1,
            "Accuracy": (total_solved + is_solved) / (total_executed + 1),
            "Time": exe_time,
            "Total Cost": Cost.instance().value,
        }
        data.append(updated_item)

        with open(result_file, 'w') as file:
            json.dump(data, file, indent=4)


if __name__ == '__main__':
    asyncio.run(main())
