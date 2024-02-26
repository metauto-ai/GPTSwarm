import pytest

import asyncio
from swarm.graph.swarm import Swarm
from swarm.environment.agents import IO, TOT
from swarm.environment.operations import DirectAnswer, FileAnalyse, Reflect, CombineAnswer, GenerateQuery, WebSearch

@pytest.mark.mock_llm
def test_run():

    input = {"task": "Who invents the AGI definition? What's the first AGI's name?", 
             "files": ["datasets/demos/agi.txt"]}
    inputs = [{"operation": "DirectAnswer", 
               "task": "Who invents the AGI definition? What's the first AGI's name?", 
               "files": ["datasets/demos/agi.txt"],
               "subtask": "Who invent AGI?",
               "output": "No one invent AGI."},
              {"operation": "FileAnalyse", 
               "task": "Who invents the AGI definition? What's the first AGI's name?", 
               "files": ["datasets/demos/agi.txt"],
               "subtask": "Read the content of ###['datasets/demos/agi.txt'], use query ###No one invent AGI.",
               "output": "However, the file content states: \"In 2050, AGI is widely used in daily life. SwarmX, a general-purpose AGI, originated from GPTSwarm.\""},
    ]

    ## Test Swarm

    swarm = Swarm(["IO", "TOT", "IO"], "gaia", model_name="mock", edge_optimize=True)
    swarm_answer = swarm.run(inputs)
    print(swarm_answer)

    ## Test Agent

    io_agent = IO(domain="gaia", model_name="mock") 
    io_agent_answer = asyncio.run(io_agent.run(inputs = inputs))
    print(io_agent_answer)

    tot_agent = TOT(domain="gaia", model_name="mock")
    tot_agent_answer = asyncio.run(tot_agent.run(inputs = inputs))
    print(tot_agent_answer)


    ## Test Operation

    da_operation = DirectAnswer(domain="gaia", model_name="mock")
    da_answer = asyncio.run(da_operation._execute(inputs)) 
    print(da_answer)

    fa_operation = FileAnalyse(domain="gaia", model_name="mock")
    fa_answer = asyncio.run(fa_operation._execute(inputs)) 
    print(fa_answer)

    reflect_operation = Reflect(domain="gaia", model_name="mock")
    reflect_answer = asyncio.run(reflect_operation._execute(inputs)) 
    print(reflect_answer)

    ca_operation = CombineAnswer(domain="gaia", model_name="mock")
    ca_answer = asyncio.run(ca_operation._execute(inputs)) 
    print(ca_answer)

    gq_operation = GenerateQuery(domain="gaia", model_name="mock")
    gq_answer = asyncio.run(gq_operation._execute(inputs)) 
    print(gq_answer)

    ws_operation = WebSearch(domain="gaia", model_name="mock")
    ws_answer = asyncio.run(ws_operation._execute(inputs)) 
    print(ws_answer)


if __name__ == "__main__":
    pytest.main([__file__])
