import pytest

from swarm.llm import VisualLLMRegistry


def test_llm_factory():
    for factory_kwargs in (
        dict(model_name='mock'),
        dict(), # default
    ):
        llm = VisualLLMRegistry.get(**factory_kwargs)
        task = "Describe this image in details."
        file_path = "datasets/demos/js.png"
        answer = llm.gen(task, file_path)
        assert isinstance(answer, str)
        assert len(answer) > 0


# TODO: test gen_video


if __name__ == "__main__":
    pytest.main()
