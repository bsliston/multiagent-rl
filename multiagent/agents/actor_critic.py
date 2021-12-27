import numpy as np
import torch
from torch import nn
from torch.distributions import categorical

from multiagent.types import BatchMemories
from multiagent.common.utils import torch_to_numpy, numpy_to_torch_float
from multiagent.agents.models import fc_model


class a2c:
    def __init__(self, input_size, output_size, device, **kwargs):
        self.input_size = input_size
        self.output_size = output_size

        self.device = device if torch.cuda.is_available() else "cpu"

        self.learning_rate = kwargs.get("learning_rate", 1e-3)
        self.gamma = kwargs.get("gamma", 0.99)

        self.reset_models()

    def reset_models(self):
        self.critic = fc_model(self.input_size, 1).to(self.device)
        self.actor = fc_model(
            self.input_size, self.output_size, categorical=True
        ).to(self.device)

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.learning_rate
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.learning_rate
        )

        self.MSELoss = nn.MSELoss()

    def train_model(self, batch_memories: BatchMemories):
        state_t = batch_memories.state_t
        action_t = batch_memories.action_t
        expected_return_t = batch_memories.expected_return_t

        # compute advantage
        value_t = self.critic(numpy_to_torch_float(state_t, device=self.device))
        expected_return = numpy_to_torch_float(
            expected_return_t, device=self.device
        )
        advantage = expected_return - value_t

        # compute critic loss
        # Critic loss becomes the MSE of the advantage calculated for the
        # critic network from the sampled memories and expected value rollouts.
        critic_loss = self.MSELoss(value_t, expected_return)

        # compute actor loss
        # Actor loss becomes the policy gradient ascent of the advantage
        # caclulated with the critic model and the actor optimizes to wrt only
        # the actor model. Goal is to maximize the gradient or minimize the
        # negative.
        at_dist = self.actor(numpy_to_torch_float(state_t, device=self.device))
        at_log_prob = at_dist.log_prob(
            numpy_to_torch_float(action_t, device=self.device)
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
