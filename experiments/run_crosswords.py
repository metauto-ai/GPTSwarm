import json
from tqdm import tqdm
import asyncio
import numpy as np
from copy import deepcopy
import pickle
import torch
import sys
import random

from swarm.environment.domain.crosswords.env import MiniCrosswordsEnv
from swarm.environment.agents.agent_registry import AgentRegistry
from swarm.graph.swarm import Swarm
from swarm.optimizer.edge_optimizer.optimization import optimize
from swarm.environment.domain.crosswords.evaluator import CrosswordsEvaluator


if __name__ == "__main__":
    if len(sys.argv) == 2:
        id = int(sys.argv[1])
        experiment_id = f"experiment{id}"
        torch.manual_seed(id)
        np.random.seed(id)
        random.seed(id)
    else:
        experiment_id = "experiment"
        id = 0
    
    print(experiment_id)

    file_path = "datasets/crosswords/mini0505_0_100_5.json"
    with open(file_path, "r") as file:
        test_data = json.load(file)

    init_connection_probability = .1
    batch_size = 20
    use_learned_order = False
    include_inner_agent_connections = True
    connect_output_nodes_to_final_node = True
    window_size = 10
    evaluator = CrosswordsEvaluator(test_data, batch_size=batch_size, metric="words", window_size=window_size, init_socre=0.4, use_init_score=True)
    swarm = Swarm(["CrosswordsReflection", "CrosswordsToT", "CrosswordsBruteForceOpt"], "crosswords", "gpt-3.5-turbo-1106", #"gpt-4-1106-preview"
                final_node_class="ReturnAll", 
                final_node_kwargs={},
                edge_optimize=True,
                init_connection_probability=init_connection_probability, 
                connect_output_nodes_to_final_node=connect_output_nodes_to_final_node, 
                include_inner_agent_connections=include_inner_agent_connections)
    optimize(swarm, evaluator, batch_size=batch_size, num_iter=11, display_freq=1, record=True,
              experiment_id=experiment_id, lr=.4, use_learned_order=use_learned_order)