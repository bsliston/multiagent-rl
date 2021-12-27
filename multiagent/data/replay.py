import numpy as np

import pdb

np.random.seed(123)


class ReplayBuffer:
    def __init__(self, **kwargs):
        self._max_memories: int = kwargs.get("max_memories", int(1e5))
        self._batch_size: int = kwargs.get("batch_size", 32)
        self.reset()

    @property
    def max_memories(self):
        return self._max_memories

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def replay_size(self):
        return len(self._replay)

    def is_enough_samples(self):
        return self.batch_size <= self.replay_size

    def reset(self):
        self._replay = []

    def sample(self):
        if not self.replay_size:
            return
        indices = np.arange(self.replay_size)
        sampled_indices = np.random.choice(indices, self.batch_size)
        return np.array([self._replay[idx] for idx in sampled_indices])

    def push(self, memory):
        self._replay.extend(memory)
        if len(self._replay) > self.max_memories:
            # Pop earlier episodes as replay becomes larger than max episodes
            # desired.
            self._replay = self._replay[-self.max_memories :]
