import pdb
from pettingzoo.atari import pong_v2
import time


def run_episode(env):
    env.reset()
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        if done:
            break
        env.step(env.action_space(agent).sample())
        env.render()
        time.sleep(0.025)


if __name__ == "__main__":
    env = pong_v2.env(num_players=2)
    number_episodes = 10

    for ep in range(1, number_episodes + 1):
        print("Episode {}".format(ep))
        run_episode(env)
