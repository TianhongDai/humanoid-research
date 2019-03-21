import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.distributions.categorical import Categorical
import random

def select_actions(pi, dist_type):
    if dist_type == 'gauss':
        mean, std = pi
        actions = Normal(mean, std).sample()
    elif dist_type == 'beta':
        alpha, beta = pi
        actions = Beta(alpha.detach().cpu(), beta.detach().cpu()).sample()
    # return actions
    return actions.detach().cpu().numpy().squeeze()

# evaluate the actions
def evaluate_actions(pi, actions, dist_type):
    if dist_type == 'gauss':
        mean, std = pi
        normal_dist = Normal(mean, std)
        log_prob = normal_dist.log_prob(actions).sum(dim=1, keepdim=True)
        entropy = normal_dist.entropy().mean()
    elif dist_type == 'beta':
        alpha, beta = pi
        beta_dist = Beta(alpha, beta)
        log_prob = beta_dist.log_prob(actions).sum(dim=1, keepdim=True)
        entropy = beta_dist.entropy().mean()
    return log_prob, entropy
