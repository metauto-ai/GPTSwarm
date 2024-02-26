import pytest

from swarm.environment.agents.cot import COT


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [
    pytest.param('mock', marks=pytest.mark.mock_llm),
    pytest.param(None),
])
async def test_cot(model_name):
    cot = COT("mmlu", model_name)
    assert cot is not None
    question = ("If event A leads to event B, and event A leads to event C, "
                "what is the relation between events B and C that "
                "human folk do not get more often than not?\n"
                "Option A: assosiation\n"
                "Option B: same event\n"
                "Option C: causation\n"
                "Option D: correlation\n"
            )
    print(question)
    response = await cot.run([{"task": question}])
    print(response)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-m", "not mock_llm"])
