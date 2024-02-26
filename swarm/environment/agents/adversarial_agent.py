from swarm.graph import Graph
from swarm.environment.operations.adversarial_answer import AdversarialAnswer
from swarm.environment.agents.agent_registry import AgentRegistry


@AgentRegistry.register('AdversarialAgent')
class AdversarialAgent(Graph):
    def build_graph(self):


        adversarial_answer = AdversarialAnswer(self.domain, self.model_name)

        self.input_nodes = [adversarial_answer]
        self.output_nodes = [adversarial_answer]

        self.add_node(adversarial_answer)
