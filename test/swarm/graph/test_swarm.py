import pytest
from typing import Optional

from swarm.graph.swarm import Swarm


@pytest.mark.parametrize("model_name", [
    pytest.param('mock', marks=pytest.mark.mock_llm),
    pytest.param(None),
])
@pytest.mark.filterwarnings("ignore:PytestAssertRewriteWarning")
def test_swarm(model_name: Optional[str]):
    task = "Tell me more about this image and summarize it in 3 sentences."
    files = ["./datasets/demos/js.png"]
    inputs = {"task": task, "files": files}

    for agent_list in (
        ["IO", "IO", "IO"],
        # ["TOT", "IO"], # cyclic dependency sometimes is here but not always
    ):
        swarm = Swarm(agent_list, "gaia", model_name=model_name, edge_optimize=True)
        answer = swarm.run(inputs)
        print(answer)
        assert isinstance(answer, list)
        assert len(answer) >= 1
        for ans in answer:
            assert isinstance(ans, str)


if __name__ == "__main__":
    pytest.main([__file__])
