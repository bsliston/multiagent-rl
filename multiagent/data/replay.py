import numpy as np

import pdb

np.random.seed(123)


class ReplayBuffer:
    def __init__(self, max_memories=10000):
        self.max_memories = max_memories
        self.reset()

    @property
    def replay_size(self):
        return len(self.replay)

    @property
    def replay(self):
        return self._replay

    def reset(self):
        self._replay = []

    def sample(self, batch_size):
        indices = np.arange(self.replay_size)
        sampled_indices = np.random.choice(indices, batch_size)
        return [self._replay[idx] for idx in sampled_indices]

    def push(self, memory):
        self._replay.append(memory)
        if len(self._replay) > self.max_memories:
            # Pop earlier episodes as replay becomes larger than max episodes
            # desired.
            self._replay = self._replay[-self.max_memories :]
