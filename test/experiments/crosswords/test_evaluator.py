import json
import asyncio
import numpy as np
import pytest

from swarm.environment.agents.agent_registry import AgentRegistry
from swarm.environment.domain.crosswords.evaluator import CrosswordsEvaluator


def evaluate(agent_str, evaluator, num_samples=None):
    agent = AgentRegistry.get(agent_str, domain="crosswords", model_name="gpt-3.5-turbo-1106")
    if num_samples is None:
        num_samples = evaluator.sample_size
    scores = []
    for _ in range(num_samples):
        scores.append(asyncio.run(evaluator.evaluate(agent)))
    print(np.array(scores).mean(), np.array(scores).std())

def test_evaluate():
    file_path = "datasets/crosswords/mini0505_0_100_5.json"
    with open(file_path, "r") as file:
        test_data = json.load(file)
    evaluator = CrosswordsEvaluator(test_data)
    # test_evaluate("CrosswordsToT", evaluator, 1)
    evaluate("CrosswordsBruteForceOpt", evaluator, 2)
    # test_evaluate("CrosswordsReflection", evaluator, 3)

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
