import pytest
import torch

from swarm.graph.composite_graph import CompositeGraph
from swarm.graph.node import Node
from swarm.optimizer.edge_optimizer.parameterization import EdgeWiseDistribution

pytestmark = pytest.mark.mock_llm


class EmptyNode(Node):
    async def _execute(self, inputs, **kwargs):
        return {"output": "test"}


@pytest.mark.parametrize("learned_order", [False, True])
@pytest.mark.parametrize("threshold, expected_edges", [(0.0, 1), (1.0, 0), (None, 0)])
def test_threshold_zero_is_deterministic(monkeypatch, learned_order, threshold, expected_edges):
    source = EmptyNode("source", "source", True)
    target = EmptyNode("target", "target", True)
    graph = CompositeGraph(target, "test")
    graph.add_node(source)
    distribution = EdgeWiseDistribution([("source", "target")], initial_probability=0.25)
    monkeypatch.setattr(torch, "rand", lambda *args: torch.tensor([0.9]))
    monkeypatch.setattr(
        distribution,
        "realize_ranks",
        lambda *args: ({"source": 0, "target": 1}, torch.tensor(0.0)),
    )

    realized, log_prob = distribution.realize(
        graph, threshold=threshold, use_learned_order=learned_order
    )

    assert realized.num_edges == expected_edges
    assert torch.isfinite(log_prob)
    assert graph.num_edges == 0
