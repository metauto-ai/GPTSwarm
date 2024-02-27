import pytest
from typing import Optional

from swarm.graph.swarm import Swarm
from swarm.environment.operations.final_decision import MergingStrategy
from experiments.evaluator.evaluator import Evaluator
from experiments.evaluator.datasets.mmlu_dataset import MMLUDataset
from datasets.MMLU.download import download


@pytest.fixture
def mmlu_data():
    download()
    return None


def make_swarm(model_name: Optional[str]) -> Swarm:
    agent_name_list = 1 * ["IO"] + 1 * ["AdversarialAgent"]
    domain = 'mmlu'
    swarm = Swarm(
        agent_name_list,
        domain,
        model_name=model_name,
        final_node_class="FinalDecision",
        final_node_kwargs=dict(strategy=MergingStrategy.RandomChoice),
        edge_optimize=True,
    )
    return swarm


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [
    pytest.param('mock', marks=pytest.mark.mock_llm),
    pytest.param(None),
])
async def test_evaluator(model_name, mmlu_data):
    swarm = make_swarm(model_name)
    dataset = MMLUDataset('dev')
    evaluator = Evaluator(swarm, dataset, dataset, model_name=model_name)
    limit_questions = 2
    score = await evaluator.evaluate_direct_answer(
        limit_questions=limit_questions)
    assert isinstance(score, float)
    score = await evaluator.evaluate_swarm(
        mode='full_connected_swarm',
        limit_questions=limit_questions)
    assert isinstance(score, float)
    score = await evaluator.evaluate_swarm(
        mode='randomly_connected_swarm',
        limit_questions=limit_questions)
    assert isinstance(score, float)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [
    pytest.param('mock', marks=pytest.mark.mock_llm),
    pytest.param(None),
])
async def test_optimization(model_name, mmlu_data):
    swarm = make_swarm(model_name)
    dataset = MMLUDataset('dev')
    evaluator = Evaluator(swarm, dataset, dataset, model_name=model_name)
    limit_questions = 2
    edge_probs = await evaluator.optimize_swarm(num_iters=2, lr=1e-1)
    assert edge_probs.numel() > 0, "Seems that edge optimization is not enabled"
    score = await evaluator.evaluate_swarm(
        mode='external_edge_probs',
        edge_probs=edge_probs,
        limit_questions=limit_questions,
        )
    assert isinstance(score, float)


def test_dataset(mmlu_data):
    for split in ('dev', 'val', 'test'):
        dataset = MMLUDataset(split)
        print(f"{split=} {len(dataset)=}")

    model_name = 'mock'
    swarm = make_swarm(model_name)
    dataset_train = MMLUDataset('dev')
    dataset_val = MMLUDataset('val')
    evaluator = Evaluator(swarm, dataset_train, dataset_val, model_name=model_name)
    size = sum(1 for _ in evaluator._train_dataset)
    assert size > 0
    size = sum(1 for _ in evaluator._val_dataset)
    assert size > 0
    for record in evaluator._train_dataset:
        correct_answer = evaluator._train_dataset.record_to_target_answer(record)
        assert isinstance(correct_answer, str), f"{record}"
    for record in evaluator._val_dataset:
        correct_answer = evaluator._val_dataset.record_to_target_answer(record)
        assert isinstance(correct_answer, str), f"{record}"

    dataset_test = MMLUDataset('test')
    for record in dataset_test:
        correct_answer = dataset_test.record_to_target_answer(record)
        assert isinstance(correct_answer, str), f"{record}"


if __name__ == "__main__":
   pytest.main([__file__, "-s", "-k", "test_dataset"])
