from typing import Dict, Optional, List
import numpy as np

from multiagent.types import BatchMemories
from multiagent.common.computations import compute_expected_return
from multiagent.common.transformations import normalize_array

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
        time: int = 1
        memories = {agent: [] for agent in self.agent_keys}
        for agent in self.env.agent_iter():
            memory = self.run_step(self.env, agent)
            if memory:
                memories[agent].append(memory)
            else:
                break

            # Assumption if iterating agent is last agent in agent keys to
            # increment episode time being tracked.
            if agent == self.agent_keys[-1]:
                time += 1

            if (time % self.update_interval == 0) and self.replay_buffers.get(
                agent
            ).is_enough_samples():
                self._train_samples(agent)

        self._add_expected_to_replay(
            {key: np.array(val) for key, val in memories.items()}
        )

    def run_step(self, env, agent):
        state_t, reward_t, done_t, info_t = env.last()
        state_t = self.transform_state(state_t)
        if done_t:
            return False

        action_t = self.agents[agent].policy(state_t)
        env.step(action_t)

        state_tp1 = env.observe(agent)
        state_tp1 = self.transform_state(state_tp1)
        action_tp1 = self.agents[agent].policy(state_tp1)

        return [
            state_t,
            action_t,
            reward_t,
            state_tp1,
            action_tp1,
            done_t,
        ]

    def _train_samples(self, agent):
        samples = self.replay_buffers.get(agent).sample()
        samples = [
            np.vstack(samples[:, replay_idx])
            for replay_idx in range(samples.shape[-1])
        ]
        samples[0] = self._normalize_state(samples[0], agent)
        samples[3] = self._normalize_state(samples[3], agent)

        batch_samples = self._get_batch_memories(samples)
        self.agents[agent].train_model(batch_samples)

    def _add_expected_to_replay(self, memories):
        for agent, memory in memories.items():
            expected_return = np.expand_dims(
                compute_expected_return(
                    memory[:, 2], self.agents.get(agent).gamma
                ),
                1,
            )
            memory = list(np.concatenate((memory, expected_return), 1))
            self.replay_buffers.get(agent).push(memory)

    def _normalize_state(self, state, agent):
        return normalize_array(
            state,
            self.transform_state(self.env.observation_space(agent).low),
            self.transform_state(self.env.observation_space(agent).high),
        )

    def transform_state(self, state):
        if self._state_transformations:
            return self._state_transformations(state)

    def _get_batch_memories(self, samples: List[np.ndarray]):
        return BatchMemories(*samples)


if __name__ == "__main__":
    pass
