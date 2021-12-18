import numpy as np
from pettingzoo.atari import pong_v2

import time
import argparse
import pdb

from actor_critic import a2c

np.random.seed(123)

import pdb


def make_env():
    return pong_v2.env(num_players=2)


def run_episode(env, network, train=True, update_interval=4, max_step=1000):
    state_t = env.reset()
    action_t = network.policy(state_t)
    episode_reward = 0.0
    episode_memories = []
    for step in range(max_step):
        state_tp1, reward_t, done, _ = env.step(action_t)
        action_tp1 = network.policy(state_tp1)

        episode_memories.append(
            [state_t, action_t, reward_t, state_tp1, action_tp1, done]
        )
        state_t = state_tp1
        action_t = action_tp1

        episode_reward += reward_t

        if train and (step + 1) % update_interval == 0:
            network.update_model()

        if done:
            break

    # get expected reward return from episodes
    expected_returns = network.get_expected_return(
        np.array(np.array(episode_memories)[:, 2], dtype=np.float)
    )

    for memory, exp_return in zip(episode_memories, expected_returns):
        memory.append(exp_return)
    network.push_memories(episode_memories)

    return episode_reward


def main(args):
    number_episodes = args.number_episodes
    batch_size = args.batch_size
    gamma = args.gamma
    learning_rate = args.learning_rate

    interval = args.evaluation_interval
    min_solution_reward = args.min_solution_reward

    env = make_env(gym_env=args.env)

    # get env variables
    observation_shape = env.observation_space.shape[0]
    action_shape = env.action_space.n

    # initialize training network
    network = a2c(
        observation_shape,
        action_shape,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
    )

    # training loop
    avg_rewards_ot = []
    start_time = time.time()
    for episode in range(number_episodes):
        avg_reward = run_episode(
            env, network, update_interval=args.update_interval
        )

        avg_rewards_ot.append(avg_reward)

        if (episode + 1) % interval == 0:
            avg_reward = np.average(avg_rewards_ot[-interval:])
            print(
                episode + 1,
                f"{avg_reward:0.02f}",
                f"Time: {time.time()-start_time:.02f}",
            )
            if avg_reward >= min_solution_reward:
                break

            start_time = time.time()

    print(
        f"Solved with final avg reward of {avg_reward} over {interval} episodes!"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Args for training lunar landar"
    )
    parser.add_argument("--env", type=str, default="CartPole-v0")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--number-episodes", type=int, default=int(5e4))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--update-interval", type=int, default=4)
    parser.add_argument("--evaluation-interval", type=int, default=100)
    parser.add_argument("--min-solution-reward", type=float, default=200.0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
