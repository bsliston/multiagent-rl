import numpy as np
import torch
from torch import nn

from multiagent.common.utils import torch_to_numpy, numpy_to_torch_float
from multiagent.agents.models import fc_model


class a2c:
    def __init__(self, input_size, output_size, device, **kwargs):
        self.input_size = input_size
        self.output_size = output_size

        self.device = device if torch.cuda.is_available() else "cpu"

        self.batch_size = kwargs.get("batch_size", 32)
        self.learning_rate = kwargs.get("learning_rate", 1e-3)
        self.gamma = kwargs.get("gamma", 0.99)

        self.reset_models()

    def reset_models(self):
        self.critic = fc_model(self.input_size, 1).to(self.device)
        self.actor = fc_model(self.input_size, self.output_size).to(self.device)

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.learning_rate
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.learning_rate
        )

        self.MSELoss = nn.MSELoss()

    def update_model(self):
        self.critic.train()
        self.actor.train()
        if self.buffer.replay_size < self.batch_size:
            return None
        sampled_memories = self.buffer.sample(self.batch_size)
        self._update_model_with_memories(sampled_memories)

    def _update_model_with_memories(self, memories):
        st = np.array([mem[0] for mem in memories])
        at = np.array([mem[1] for mem in memories])
        rt = np.array([mem[2] for mem in memories])
        stp1 = np.array([mem[3] for mem in memories])
        atp1 = np.array([mem[4] for mem in memories])
        done = np.array([mem[5] for mem in memories])
        expected_return = np.array([mem[6] for mem in memories])

        # compute advantage
        V_t = self.critic(numpy_to_torch_float(st, device=self.device))
        expected_return = numpy_to_torch_float(
            expected_return, device=self.device
        ).unsqueeze(1)
        advantage = expected_return - V_t

        # compute critic loss
        # Critic loss becomes the MSE of the advantage calculated for the
        # critic network from the sampled memories and expected value rollouts.
        critic_loss = self.MSELoss(V_t, expected_return)

        # compute actor loss
        # Actor loss becomes the policy gradient ascent of the advantage
        # caclulated with the critic model and the actor optimizes to wrt only
        # the actor model. Goal is to maximize the gradient or minimize the
        # negative.
        at_dist = self.actor(numpy_to_torch_float(st, device=self.device))
        at_log_prob = at_dist.log_prob(
            numpy_to_torch_float(at, device=self.device)
        )
        actor_loss = (-1.0 * at_log_prob * advantage.detach()).mean()

        # update model
        loss = actor_loss + critic_loss
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        loss.backward()

        self.critic_optimizer.step()
        self.actor_optimizer.step()

    def policy(self, state):
        a_dist = self.actor(numpy_to_torch_float(state, device=self.device))
        return torch_to_numpy(a_dist.sample())

    def get_expected_return(self, rewards):
        n_rewards = rewards.size
        expected_returns = np.zeros_like(rewards)
        expected_return = 0.0

        # Expected return works backwards from last seen reward to get Gt value
        # expectation
        for ri, reward in enumerate(rewards[::-1]):
            expected_return = reward + (expected_return * self.gamma)
            expected_returns[n_rewards - ri - 1] = expected_return

        # Normalize expected returns per episode for stability during advantage
        # training.
        expected_returns = (expected_returns - np.average(expected_returns)) / (
            np.std(expected_returns) + 1e-8
        )

        return expected_returns
