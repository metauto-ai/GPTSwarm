import pytest

from swarm.environment.prompt import PromptSetRegistry


def test_prompt_set_registry():
    for name in PromptSetRegistry.registry:
        prompt_set = PromptSetRegistry.get(name)
        assert prompt_set is not None


if __name__ == "__main__":
    pytest.main([__file__])
