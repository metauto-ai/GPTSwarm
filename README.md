[![Page](https://img.shields.io/badge/Project-Page-lightgreen.svg)](https://gptswarm.org)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-gold.svg)](https://arxiv.org/abs/2402.16823)
[![License](https://img.shields.io/badge/License-MIT-orange.svg)](https://github.com/mczhuge/GPTSwarm/blob/main/LICENSE)
[![Issues](https://img.shields.io/github/issues/metauto-aiGPTSwarm?color=00afaa)](https://github.com/metauto-ai/gptswarm/issues)
[![Twitter Follow](https://img.shields.io/twitter/follow/AI_KAUST?style=social)](https://twitter.com/AI_KAUST)
[![Wechat](https://img.shields.io/badge/Wechat-7BB32E?logo=wechat&logoColor=white)](https://metauto.ai/images/wechat.jpeg)

<p align="left">
<a href=""><img src="swarm/utils/assets/logo.png" alt="GPTSwarm" width="430px"></a>
</p>

🐝 **GPTSwarm is a graph-based framework for LLM-based agents, providing two high-level features:**

* It lets you build LLM-based agents from graphs.
* It enables the customized and automatic self-organization of agent swarms with self-improvement capabilities.

## News

🚀 Feb 27, 2024: Our academic paper: [Language Agents as Optimizable Graphs](https://arxiv.org/abs/2402.16823) is released.

## Edge optimization example

Here is the edge optimization process that updates edge probabilities toward improvement of the benchmark score. Notice that within an agent, the edges are fixed, whereas the inter-agent connections are getting optimized towards either edge pruning (value 0, blue) or creation (value 1, red).

<img src="assets/edge_opt.gif" alt="Edge optimization" width="300">

## About GPTSwarm

<img src="assets/gpt_swarm.png" alt="Framework" width="799">

At a granular level, GPTSwarm is a library that includes the following components: 


| Module | Description |
| ---- | --- |
| [**swarm.environment**](swarm/environment) | Domain-specific operations, agents, tools, and tasks |
| [**swarm.graph**](swarm/graph) | Graph-related functions for creating and executing agent graphs and swarm composite graphs |
| [**swarm.llm**](swarm/llm) | Interface for selecting LLM backends and calculating their operational costs |
| [**swarm.memory**](swarm/memory) | Index-based memory |
| [**swarm.optimizer**](swarm/optimizer) | Optimization algorithms designed to enhance agent performance and overall swarm efficiency |


## Quickstart

**Clone the repo**

```bash
git clone --recurse-submodules https://github.com/mczhuge/GPTSwarm.git
cd GPTSwarm/
```

**Install packages**
```
conda create -n swarm python=3.10
conda activate swarm
pip install -r requirements_py310_<linux|macos>.txt
```

**You should add API keys in `.env.template` and change its name to `.env`**

```python
OPENAI_API_KEY="" # for OpenAI LLM backend
SEARCHAPI_API_KEY="" # for Web Search
```

**Getting started with GPTSwarm is easy. Quickly run a predefined swarm**

```python
from swarm.graph.swarm import Swarm

swarm = Swarm(["IO", "IO", "IO"], "gaia")
task = "What is the capital of Jordan?"
inputs = {"task": task}
answer = await swarm.arun(inputs)
```

**or make use of tools, such as the file analyzer**

```python
from swarm.graph.swarm import Swarm
swarm = Swarm(["IO", "TOT"], "gaia")
task = "Tell me more about this image and summarize it in 3 sentences."
files = ["./datasets/demos/js.png"]
inputs = {"task": task, "files": files}
danswer = swarm.run(inputs)
```

Check out the minimal Swarm example in Colab here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mczhuge/GPTSwarm/blob/main/notebooks/demo_swarm.ipynb).

See how to create a custom Agent and run a Swarm with it here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mczhuge/GPTSwarm/blob/main/notebooks/demo_custom_agent.ipynb).

Watch the following for an example of how to run the notebooks.

https://github.com/mczhuge/GPTSwarm/assets/100462607/a05f78b7-5f9d-4762-8337-973702ce4493


**🛠 See [our experiments](https://github.com/mczhuge/GPTSwarm/tree/main/experiments) for more advanced use of our framework.**

## Class diagram

<img src="assets/class_diagram.png" alt="Edge optimization" width="700">

## Example of the Swarm

<img src="assets/swarm_v3.png" alt="Edge optimization" width="500">

## More Visualizations

<img src="assets/swarm_vis.png" alt="Edge optimization" width="800">

## Running with a local LLM

We support local LM inference via [LM Studio](https://lmstudio.ai). Download their desktop app for Mac or Windows, choose a model from the Huggingface repository and start the server. Use `model_name='lmstudio'` in GPTSwarm code to run with the local LLM.

<img src="assets/lm_studio.png" alt="Edge optimization" width="800">

## Initial Contributors

* [Mingchen Zhuge](http://metauto.ai) (PhD Student, KAUST; Project Initiator)
* [Wenyi Wang](https://scholar.google.com/citations?user=79ODhuQAAAAJ&hl=en&oi=ao) (PhD Student, KAUST; Initial Participant)
* [Dmitrii Khizbullin](http://www.khizbullin.tech) (Research Engineer Lead, KAUST; Project Engineer Lead)
* [Louis Kirsch](http://louiskirsch.com) (PhD Student, IDSIA)
* [Francesco Faccio](https://scholar.google.com/citations?user=0z3DkrkAAAAJ&hl=en&oi=ao) (PostDoc, IDSIA; Visiting Researcher, KAUST)
* [Jürgen Schmidhuber](http://www.idsia.ch/~juergen/) (Director, KAUST AI Initiative; Scientific Director, IDSIA)

Please read our [developer document](DEVELOPMENT.md) if you are interested in contributing.


## Citation
Please cite our paper if you find the library useful or interesting.
```
@misc{zhuge2024language,
      title={Language Agents as Optimizable Graphs}, 
      author={Mingchen Zhuge and Wenyi Wang and Louis Kirsch and Francesco Faccio and Dmitrii Khizbullin and Jurgen Schmidhuber},
      year={2024},
      eprint={2402.16823},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```




