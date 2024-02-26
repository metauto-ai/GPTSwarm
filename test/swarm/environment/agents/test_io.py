import pytest

from swarm.environment.agents.io import IO


@pytest.mark.parametrize("model_name", [
    pytest.param('mock', marks=pytest.mark.mock_llm),
    pytest.param(None),
])
@pytest.mark.asyncio
async def test_io(model_name):
    io = IO("gaia", model_name)
    assert io is not None
    response = await io.run([{"task": "say hello"}])
    print(response)


if __name__ == "__main__":
    pytest.main([__file__])
