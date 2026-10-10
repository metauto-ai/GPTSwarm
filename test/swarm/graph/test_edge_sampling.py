import random

import pytest

from swarm.graph.composite_graph import CompositeGraph
from swarm.graph.node import Node
from swarm.optimizer.edge_optimizer.parameterization import EdgeWiseDistribution


class DummyNode(Node):
    def __init__(self, node_id):
        super().__init__("test node", node_id, combine_inputs_as_one=False)

    async def _execute(self, inputs, **kwargs):
        return inputs


@pytest.fixture
def graph():
    graph = CompositeGraph(DummyNode("output"), "test")
    graph.add_node(DummyNode("a"))
    graph.add_node(DummyNode("b"))
    return graph


@pytest.fixture
def bounded_sampling(monkeypatch):
    """Fail promptly if an impossible edge budget causes unbounded resampling."""
    calls = 0
    sample = random.Random(0).sample

    def bounded_sample(population, k):
        nonlocal calls
        calls += 1
        assert calls <= max(1, len(population)), "repeated sampling cannot satisfy this edge budget"
        return sample(population, k)

    monkeypatch.setattr(random, "sample", bounded_sample)


@pytest.mark.mock_llm
def test_sample_requested_edges_without_mutating_source(graph):
    distribution = EdgeWiseDistribution([("a", "b"), ("b", "output")])

    sampled = distribution.random_sample_num_edges(graph, 2)

    assert sampled.num_edges == 2
    assert graph.num_edges == 0


@pytest.mark.mock_llm
@pytest.mark.parametrize("connections", [[], [("a", "b")], [("a", "b"), ("b", "a")]])
def test_unreachable_edge_budget_raises(graph, bounded_sampling, connections):
    distribution = EdgeWiseDistribution(connections)

    with pytest.raises(ValueError, match="edge"):
        distribution.random_sample_num_edges(graph, 2)


@pytest.mark.mock_llm
def test_existing_edges_already_satisfy_budget(graph):
    graph.nodes["a"].add_successor(graph.nodes["output"])
    distribution = EdgeWiseDistribution([])

    sampled = distribution.random_sample_num_edges(graph, 1)

    assert sampled.num_edges == 1
