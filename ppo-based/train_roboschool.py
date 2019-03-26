from arguments import get_args
from ppo_agent import ppo_agent
from models import MLP_Net
import gym
import os

class rew_scale(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale
    def reward(self, r):
        return r * self.scale


if __name__ == '__main__':
    # set some environment variables
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the arguments
    args = get_args()
    # make environment
    env = gym.make(args.env_name)
    network = MLP_Net(env.observation_space.shape[0], env.action_space.shape[0], args.dist)
    ppo_trainer = ppo_agent(env, args, network)
    ppo_trainer.learn()
