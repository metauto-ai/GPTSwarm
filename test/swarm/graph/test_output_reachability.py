import asyncio
import sys

import pytest

from swarm.graph.composite_graph import CompositeGraph
from swarm.graph.node import Node

pytestmark = pytest.mark.mock_llm


class RecordingNode(Node):
    def __init__(self, node_id, executed):
        super().__init__("record execution", node_id, combine_inputs_as_one=True)
        self.executed = executed

    async def _execute(self, inputs, **kwargs):
        self.executed.append(self.id)
        return {"output": self.id}


def test_shared_ancestors_execute_once_for_multiple_outputs():
    executed = []
    first = RecordingNode("first", executed)
    second = RecordingNode("second", executed)
    source = RecordingNode("source", executed)
    graph = CompositeGraph(first, "test")
    graph.add_node(second)
    graph.add_node(source)
    graph.output_nodes.append(second)
    graph.input_nodes = [source]
    source.add_successor(first)
    source.add_successor(second)
    graph.add_node(RecordingNode("unused", executed))

    assert asyncio.run(graph.run({"task": "test"})) == ["first", "second"]
    assert executed == ["source", "first", "second"]


def test_deep_graph_does_not_require_recursive_reachability():
    executed = []
    output = RecordingNode("output", executed)
    graph = CompositeGraph(output, "test")
    previous = output
    for i in range(sys.getrecursionlimit() + 10):
        node = graph.add_node(RecordingNode(str(i), executed))
        node.add_successor(previous)
        previous = node
    graph.input_nodes = [previous]

    assert asyncio.run(graph.run({"task": "test"})) == ["output"]
    assert len(executed) == len(graph.nodes)


def test_dead_diamond_branches_do_not_revisit_every_path():
    executed = []
    visits = 0

    class CountedSuccessors(list):
        def __iter__(self):
            nonlocal visits
            visits += 1
            assert visits <= 100, "reachability repeatedly walks the same dead branches"
            return super().__iter__()

    output = RecordingNode("output", executed)
    graph = CompositeGraph(output, "test")
    graph.input_nodes = [output]
    previous_layer = []
    for depth in range(10):
        layer = [graph.add_node(RecordingNode(f"{depth}-{i}", executed)) for i in range(2)]
        for source in previous_layer:
            for target in layer:
                source.add_successor(target)
        previous_layer = layer
    for node in graph.nodes.values():
        node.successors = CountedSuccessors(node.successors)

    assert asyncio.run(graph.run({"task": "test"})) == ["output"]
    assert executed == ["output"]


def test_graph_without_outputs_executes_no_nodes():
    executed = []
    graph = CompositeGraph(RecordingNode("unused", executed), "test")
    graph.output_nodes = []

    assert asyncio.run(graph.run({})) == ["No answer since there are no inputs provided"]
    assert executed == []
