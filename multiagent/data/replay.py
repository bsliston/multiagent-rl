import numpy as np

import pdb

np.random.seed(123)


class ReplayBuffer:
    def __init__(self, agents, max_memories=10000):
        self._agents = agents
        self.max_memories = max_memories
        self.reset()

    @property
    def replay_size(self):
        return len(self.replay)

    @property
    def agents(self):
        return self._agents

    def reset(self):
        self.replay = []

    def sample(self, batch_size):
        indices = np.arange(self.replay_size)
        sampled_indices = np.random.choice(indices, batch_size)
        return [self.replay[idx] for idx in sampled_indices]

    def push(self, memory):
        self.replay.append(memory)
        if len(self.replay) > self.max_memories:
            # Pop earlier episodes as replay becomes larger than max episodes
            # desired.
            self.replay = self.replay[-self.max_memories :]
