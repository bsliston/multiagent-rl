import abc
from typing import Dict, Optional
import numpy as np

import pdb


class AgentTrainer:
    def __init__(
        self,
        agents,
        env,
        replay_buffers,
        update_interval=10,
        state_transformations=None,
    ):
        self._agents = agents
        self._env = env
        self._replay_buffers = replay_buffers
        self.update_interval = update_interval
        self._state_transformations = state_transformations
        self.time: int = 1

    @property
    def agents(self):
        return self._agents

    @property
    def agent_keys(self):
        return list(self.agents.keys())

    @property
    def env(self):
        return self._env

    @property
    def replay_buffers(self):
        return self._replay_buffers

    def train_episode(self):
        self.env.reset()
        for agent in self.env.agent_iter():
            self.run_step(self.env, agent)

            # Assumption if iterating agent is last agent in agent keys to
            # increment episode time being tracked.
            if agent == self.agent_keys[-1]:
                self.time += 1

            if self.time % self.update_interval == 0:
                self._train_samples(agent)

    def run_step(self, env, agent):
        state_t, reward_t, done_t, info_t = env.last()
        if done_t:
            return

        # replace this with policy
        action_t = env.action_space(agent).sample()
        env.step(action_t)

        state_tp1 = env.observe(agent)
        action_tp1 = env.action_space(agent).sample()

        self._add_to_replay(
            agent,
            [
                self.transform_state(state_t),
                action_t,
                reward_t,
                self.transform_state(state_tp1),
                action_tp1,
                done_t,
            ],
        )

    def _train_samples(self, agent):

        return NotImplemented

    def _add_to_replay(self, agent, memory):
        self.replay_buffers.get(agent).push(memory)

    def transform_state(self, state):
        if self._state_transformations:
            return self._state_transformations(state)


if __name__ == "__main__":
    pass
