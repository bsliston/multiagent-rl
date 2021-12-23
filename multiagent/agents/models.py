import torch
from torch.nn import functional as F
from torch.distributions import Categorical

torch.manual_seed(123)

import pdb


class fc_model(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden=128):
        super(fc_model, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.fc3 = torch.nn.Linear(hidden, output_size)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, inp):
        x = F.relu(self.fc1(inp))
        x = F.relu(self.fc2(x))
        return Categorical(F.softmax(self.fc3(x), dim=-1))

    def save_weights(self, path):
        return NotImplemented

    def load_weights(self, path):
        return NotImplemented
