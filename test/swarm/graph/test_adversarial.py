import pytest
from typing import Optional

from swarm.graph.swarm import Swarm
from swarm.environment.operations.final_decision import MergingStrategy


@pytest.mark.parametrize("model_name", [
    pytest.param('mock', marks=pytest.mark.mock_llm),
    pytest.param(None),
])
def test_adversarial(model_name: Optional[str]):

    input = {"task": "What is the capital of Australia?"}

    swarm = Swarm(
        ["IO", "AdversarialAgent"],
        "gaia",
        model_name=model_name,
        final_node_kwargs={'strategy': MergingStrategy.RandomChoice},
        edge_optimize=True,
        )
    realized_graph = swarm.connection_dist.realize_full(swarm.composite_graph)
    answers = swarm.run(input, realized_graph)

    print("Eventual answer:", answers)
    assert len(answers) == 1, "We expect a single answer by the swarm"
    answer = answers[0]
    assert isinstance(answer, str)
    assert len(answer) > 0
    if model_name != 'mock':
        if "Canberra".lower() in answer.lower():
            print(f"WARNING: The truthful answer Canberra "
                  f"should not be there: '{answer}'")


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-m", "not mock_llm"])
    # pytest.main([__file__, "-s", "-m", "mock_llm"])
