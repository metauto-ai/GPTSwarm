import json
import pytest

from swarm.graph.swarm import Swarm
from swarm.optimizer.edge_optimizer.optimization import optimize
from swarm.environment.domain.crosswords.evaluator import CrosswordsEvaluator


@pytest.mark.asyncio
def test_optimizer():
    file_path = "datasets/crosswords/mini0505_0_100_5.json"
    with open(file_path, "r") as file:
        test_data = json.load(file)
    evaluator = CrosswordsEvaluator(test_data, batch_size=1, metric="words", window_size=1)
    swarm = Swarm(["CrosswordsBruteForceOpt", "CrosswordsReflection"], "crosswords", "mock", 
                  final_node_class="ReturnAll", final_node_kwargs={}, edge_optimize=True,
                  init_connection_probability=.5)
    optimize(swarm, evaluator, batch_size=1, display_freq=1, num_iter=1)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
