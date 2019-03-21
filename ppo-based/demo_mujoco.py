from arguments import get_args
from ppo_agent import ppo_agent
import numpy as np
from models import MLP_Net
import torch
import os
import gym
import cv2

# get tensors
def get_tensors(obs):
    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

# normalize
def normalize(x, mean, std, clip=10):
    return np.clip((x - mean) / std, -clip, clip)

if __name__ == '__main__':
    args = get_args()
    # create environment
    env = gym.make(args.env_name)
    # start to create model
    model_path = args.save_dir + args.env_name + '/model.pt'
    network = MLP_Net(env.observation_space.shape[0], env.action_space.shape[0], args.dist)
    network_model, obs_mean, obs_std = torch.load(model_path, map_location=lambda storage, loc: storage)
    network.load_state_dict(network_model)
    obs = normalize(env.reset(), obs_mean, obs_std)
    reward_total = 0
    for _ in range(10000):
        env.render()
        obs_tensor = get_tensors(obs)
        with torch.no_grad():
            _, pi = network(obs_tensor)
        # select actions
        if args.dist == 'gauss':
            mean, std = pi
            actions = mean.detach().cpu().numpy().squeeze()
        elif args.dist == 'beta':
            alpha, beta = pi
            actions = (alpha - 1) / (alpha + beta - 2)
            actions = actions.detach().cpu().numpy().squeeze()
            actions = -1 + 2 * actions 
        obs, reward, done, _ = env.step(actions)
        reward_total += reward
        if done:
            break
        obs = normalize(obs, obs_mean, obs_std)
    print('the total reward in this episode is {}'.format(reward_total))
