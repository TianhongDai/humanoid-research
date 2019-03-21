import torch
from torch import nn
from torch.nn import functional as F

"""
this network also include gaussian distribution and beta distribution

"""

class MLP_Net(nn.Module):
    def __init__(self, state_size, num_actions, dist_type):
        super(MLP_Net, self).__init__()
        self.dist_type = dist_type
        self.fc1_v = nn.Linear(state_size, 64)
        self.fc2_v = nn.Linear(64, 64)
        self.fc1_a = nn.Linear(state_size, 64)
        self.fc2_a = nn.Linear(64, 64)

        # check the type of distribution
        if self.dist_type == 'gauss':
            self.sigma_log = nn.Parameter(torch.zeros(1, num_actions))
            self.action_mean = nn.Linear(64, num_actions)
            self.action_mean.weight.data.mul_(0.1)
            self.action_mean.bias.data.zero_()
        elif self.dist_type == 'beta':
            self.action_alpha = nn.Linear(64, num_actions)
            self.action_beta = nn.Linear(64, num_actions)
            # init..
            self.action_alpha.weight.data.mul_(0.1)
            self.action_alpha.bias.data.zero_()
            self.action_beta.weight.data.mul_(0.1)
            self.action_beta.bias.data.zero_()

        # define layers to output state value
        self.value = nn.Linear(64, 1)
        self.value.weight.data.mul_(0.1)
        self.value.bias.data.zero_()

    def forward(self, x):
        x_v = torch.tanh(self.fc1_v(x))
        x_v = torch.tanh(self.fc2_v(x_v))
        state_value = self.value(x_v)
        # output the policy...
        x_a = torch.tanh(self.fc1_a(x))
        x_a = torch.tanh(self.fc2_a(x_a))
        if self.dist_type == 'gauss':
            mean = self.action_mean(x_a)
            sigma_log = self.sigma_log.expand_as(mean)
            sigma = torch.exp(sigma_log)
            pi = (mean, sigma)
        elif self.dist_type == 'beta':
            alpha = F.softplus(self.action_alpha(x_a)) + 1
            beta = F.softplus(self.action_beta(x_a)) + 1
            pi = (alpha, beta)
        return state_value, pi
