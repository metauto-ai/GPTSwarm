import pytest
import json
import asyncio

from swarm.environment.domain.crosswords.env import MiniCrosswordsEnv
from swarm.environment.agents.agent_registry import AgentRegistry


@pytest.mark.asyncio
async def test():
    file_path = "datasets/crosswords/mini0505_0_100_5.json"
    with open(file_path, "r") as file:
        data = json.load(file)
    env = MiniCrosswordsEnv(data)
    env.reset(0)

    inputs = {"env": env}
    agent = AgentRegistry.get('CrosswordsReflection', domain="crosswords", model_name="gpt-3.5-turbo-1106")
    answer = await agent.run(inputs, max_tries=1)
    print(answer[0]['env'].render())


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
