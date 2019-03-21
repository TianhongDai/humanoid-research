from arguments import get_args
from ppo_agent import ppo_agent
from models import MLP_Net
import gym
import os

class RewScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale
    def reward(self, r):
        return r * self.scale


if __name__ == '__main__':
    args = get_args()
    # make environment
    env = gym.make(args.env_name)
    env = RewScale(env, 0.1)
    network = MLP_Net(env.observation_space.shape[0], env.action_space.shape[0], args.dist)
    ppo_trainer = ppo_agent(env, args, network)
    ppo_trainer.learn()
