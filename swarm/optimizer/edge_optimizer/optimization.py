import torch
import torch.nn as nn
from tqdm import tqdm
import asyncio
import pickle
import numpy as np

def optimize(swarm, evaluator, num_iter=100, lr=1e-1, display_freq=10, batch_size=4, record=False, experiment_id='experiment', use_learned_order=False):
    optimizer = torch.optim.Adam(swarm.connection_dist.parameters(), lr=lr)
    pbar = tqdm(range(num_iter))
    utilities = []
    loop = asyncio.get_event_loop()
    for step in pbar:
        evaluator.reset()
        optimizer.zero_grad()
        tasks = []
        log_probs = []
        for i in range(batch_size):
            _graph, log_prob = swarm.connection_dist.realize(swarm.composite_graph, use_learned_order=use_learned_order)
            tasks.append(evaluator.evaluate(_graph, return_moving_average=True))
            log_probs.append(log_prob)
        results = loop.run_until_complete(asyncio.gather(*tasks))
        utilities.extend([result[0] for result in results])
        if step == 0:
            moving_averages = np.array([np.mean(utilities) for _ in range(batch_size)])
        else:
            moving_averages = np.array([result[1] for result in results])
        loss = (-torch.stack(log_probs) * torch.tensor(np.array(utilities[-batch_size:]) - moving_averages)).mean()
        loss.backward()
        optimizer.step()

        if i % display_freq == display_freq - 1:
            print(f'avg. utility = {np.mean(utilities[-batch_size:]):.3f} with std {np.std(utilities[-batch_size:]):.3f}')
            if record:
                with open(f"result/crosswords/{experiment_id}_utilities_{step}.pkl", "wb") as file:
                    pickle.dump(utilities, file)
                torch.save(swarm.connection_dist.state_dict(), f"result/crosswords/{experiment_id}_edge_logits_{step}.pt")