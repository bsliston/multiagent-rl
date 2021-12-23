from typing import Tuple
import numpy as np
from pettingzoo.atari import pong_v2

from multiagent.trainer import AgentTrainer
from multiagent.data.replay import ReplayBuffer
from multiagent.common.transformations import state_flatten

from multiagent.agents.actor_critic import a2c

import time
import argparse
import pdb


def make_env():
    return pong_v2.env(num_players=2)


def initialize_agents(env, device, batch_size, learning_rate, gamma):
    agents = {}
    for agent_name in env.possible_agents:
        observation_space = np.prod(env.observation_space(agent_name).shape)
        action_space = env.action_space(agent_name).n
        agents[agent_name] = a2c(
            observation_space,
            action_space,
            device,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gamm=gamma,
        )
    return agents


def initialize_replay(env, max_memories):
    agent_replays = {}
    for agent_name in env.possible_agents:
        agent_replays[agent_name] = ReplayBuffer(max_memories=max_memories)
    return agent_replays


def main(args):
    env = make_env()
    agents = initialize_agents(
        env,
        args.device,
        args.batch_size,
        args.learning_rate,
        args.gamma,
    )

    replay_buffers = initialize_replay(env, args.max_replay)

    trainer = AgentTrainer(
        agents,
        env,
        replay_buffers,
        update_interval=args.update_interval,
        state_transformations=state_flatten,
    )

    for episode in range(args.number_episodes):
        print("Episode {}".format(episode))
        trainer.train_episode()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for training pong environment"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-replay", type=int, default=int(5e4))
    parser.add_argument("--update-interval", type=int, default=4)
    parser.add_argument("--number-episodes", type=int, default=int(1e5))

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
