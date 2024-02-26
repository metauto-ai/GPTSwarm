import pytest
from typing import Optional

from swarm.llm import LLMRegistry, Message


@pytest.mark.parametrize("model_name", [
    pytest.param('mock', marks=pytest.mark.mock_llm),
    pytest.param(None),
])
def test_llm_factory(model_name: Optional[str]):
    llm = LLMRegistry.get(model_name)
    message = Message(role='user', content="What is the capital of Australia?")
    answer = llm.gen([message])
    assert isinstance(answer, str)
    assert len(answer) > 0


@pytest.mark.parametrize("model_name", [
    pytest.param('mock', marks=pytest.mark.mock_llm),
    pytest.param(None),
])
@pytest.mark.asyncio
async def test_llm_factory_async(model_name: Optional[str]):
    llm = LLMRegistry.get(model_name)
    message = Message(role='user', content="What is the capital of Australia?")
    answer = await llm.agen([message])
    assert isinstance(answer, str)
    assert len(answer) > 0


if __name__ == "__main__":
    pytest.main()
