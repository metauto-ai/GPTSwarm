from copy import deepcopy
import numpy as np
import random

from swarm.environment.domain.crosswords.env import MiniCrosswordsEnv


class CrosswordsEvaluator():
    def __init__(self, data, batch_size=4, metric="words", window_size=10, init_socre=.5, use_init_score=False):
        self.env = MiniCrosswordsEnv(data)
        self.sample_size = len(data)
        self.batch_size = batch_size
        self.metric = metric
        self.window_size = window_size
        self.init_score = init_socre
        self.use_init_score = use_init_score
        self.reset()

    @property
    def moving_average(self):
        return np.mean(self.scores[-10:])
    
    def reset(self):
        self.scores = [[] for _ in range(self.sample_size)]
        self.idx = self.sample_size - 1

    def shuffle_data(self):
        self.idx = 0
        self.perm = np.random.permutation(self.sample_size)

    async def evaluate(self, graph, return_moving_average=False):
        self.idx += 1
        if self.idx == self.sample_size:
            self.shuffle_data()
        problem_idx = self.perm[self.idx]
        env = deepcopy(self.env)
        env.reset(self.perm[problem_idx])
        inputs = {"env": env}
        answer = (await graph.run(inputs, max_time=10000, max_tries=1, return_all_outputs=True))
        if not isinstance(answer, list):
            Warning("Answer is not a dictionary")
            score = 0
        else:
            answer = max(answer, key=lambda x: getattr(x['env'], f'r_{self.metric[:-1]}'))
            if self.metric == "letters":
                score = answer['env'].r_letter
            elif self.metric == "words":
                score = answer['env'].r_word
            else:
                score = answer['env'].r_game

        self.scores[problem_idx].append(score)
        if return_moving_average:
            if len(self.scores[problem_idx]) == 1 or self.use_init_score:
                return score, self.init_score
            return score, np.mean(self.scores[problem_idx][-self.window_size:])
        return score