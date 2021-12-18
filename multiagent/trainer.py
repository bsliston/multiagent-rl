import abc
from typing import Dict, Optional


class AgentTrainer:
    def __init__(self, agents, env):
        self._agents = agents
        self._env = env

    def train_episode(self):
        pass

    def run_episode(self):
        return NotImplemented

    def _train_samples(self):
        pass

    def _add_to_replay(self):
        pass


if __name__ == "__main__":
    pass
