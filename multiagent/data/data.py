import numpy as np

import pdb

np.random.seed(123)


class replay_buffer:
    def __init__(self, max_memories=10000):
        self.max_memories = max_memories
        self.reset()

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

    def reset(self):
        self.replay = []

    @property
    def replay_size(self):
        return len(self.replay)
