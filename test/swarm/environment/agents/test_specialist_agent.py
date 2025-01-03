import pytest

from swarm.environment.agents.mmlu.specialist_agent import SpecialistAgent


@pytest.mark.parametrize("model_name", [
    pytest.param('mock', marks=pytest.mark.mock_llm),
    pytest.param(None),
])
@pytest.mark.asyncio
async def test_io(model_name):
    task = """
What is love?
A: a feeling
B: all you need
C: a chemical process
D: baby don't hurt me no more
"""
    responses = []
    for _ in range(10):
        io = SpecialistAgent("mmlu", model_name)
        response = await io.run({"task": task})
        print(response[0])
        responses.append(response[0])

    print(responses)
    print()

    print([r[0] for r in responses])
    print()


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-m", "not mock_llm"])
