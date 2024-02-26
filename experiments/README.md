## Run the following commands to reproduce our experiments in the paper

### Download dataset

The datasets are stored in git LFS. If git LFS is already installed in your system, no action is required. To check if git LFS is installed, run:
```bash
git lfs
```
If you see an error message, install git LFS via this instruction](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage). Then run:
```bash
git lfs install
git lfs pull
```

### Include necessary submodules

If the project has already been cloned:
```bash
git submodule init
git submodule update
```
or a one-liner:
```bash
git submodule update --init --recursive
```

### **MMLU**
Run the baseline:
```bash
PYTHONPATH=. python experiments/run_mmlu.py --mode=DirectAnswer
```

Run fully-connected swarm ablation:
```bash
PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=3 --mode=FullConnectedSwarm
```

Run randomly-connected swarm ablation:
```bash
PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=3 --mode=RandomSwarm
```

Run the main experiment with optimization and eventual evaluation:
```bash
PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=3 --mode=OptimizedSwarm
```

### **Mini Crosswords**
Run the REINFORCE algorithm for edge optimization with three agents as described in the paper.
```bash
PYTHONPATH=. python experiments/run_crosswords.py
```

### **HumanEval**
Run node optimization that improves the demonstration examples of each node.
```bash
PYTHONPATH=. python experiments/run_humaneval.py  --learn_demonstration True
```

### **GAIA**
Run the general assistant tasks.
```bash
PYTHONPATH=. python experiments/run_gaia.py
```