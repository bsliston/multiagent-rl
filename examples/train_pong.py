from typing import Tuple
import numpy as np
from pettingzoo.atari import pong_v2

from multiagent.trainer import AgentTrainer

from multiagent.agents.actor_critic import a2c

import time
import argparse
import pdb


def make_env():
    return pong_v2.env(num_players=2)


def initialize_agents(env, device):
    agents = {}
    for agent_name in env.possible_agents:
        observation_space = np.prod(env.observation_space(agent_name).shape)
        action_space = env.action_space(agent_name).n
        agents[agent_name] = a2c(observation_space, action_space, device)
    return agents


def main(args):
    env = make_env()
    agents = initialize_agents(env, args.device)
    trainer = AgentTrainer(agents, env)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for training pong environment"
    )
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--number-episodes", type=int, default=int(5e4))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--update-interval", type=int, default=4)
    parser.add_argument("--evaluation-interval", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
