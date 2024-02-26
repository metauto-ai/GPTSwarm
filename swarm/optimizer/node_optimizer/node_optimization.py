import random
import asyncio

from swarm.environment.operations.optimizable_operation import OptimizableOperation

async def optimize(node: OptimizableOperation, learn_demonstration=False, learn_prompt=True):
    examples = node.memory.query_by_id(node.id)[-4:] 
    positive_examples = [example for example in examples if node.memory.query_by_id(example['task'])[0]]
    negative_examples = [example for example in examples if not node.memory.query_by_id(example['task'])[0]]

    prompts = [node.prompt]
    demonstrations = [node.domenstrations]
    if learn_demonstration:
        new_domenstrations = node.domenstrations + positive_examples
        if len(new_domenstrations) > node.max_domenstrations:
            new_domenstrations = random.sample(new_domenstrations, node.max_domenstrations)
        demonstrations.append(new_domenstrations)
    
    if learn_prompt and len(negative_examples) > 0:
        new_prompt = await node.get_new_prompt(negative_examples)
        prompts.append(new_prompt)

    candidates = [(prompt, domenstrations) for prompt in prompts for domenstrations in demonstrations]
    if len(candidates) == 1:
        return
    tasks = [node.evaluate(candidate) for candidate in candidates]
    scores = await asyncio.gather(*tasks)
    node.prompt, node.domenstrations = candidates[scores.index(max(scores))]

