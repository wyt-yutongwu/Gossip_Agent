import os
import sys
import numpy as np
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import pandas as pd

import argparse
import torch
from ssd.envs.env_creator import create_env
from DQN_model.multi_agent import mult_agent_dqn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser("Stable-Baselines3 PPO with Parameter Sharing")
    parser.add_argument(
        "--env-name",
        type=str,
        default="harvest",
        choices=["harvest", "cleanup","harvest_test","harvest_paper", "harvest_simple","harvest_sa"],
        help="The SSD environment to use",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=10,
        help="The number of agents",
    )
    parser.add_argument(
        "--rollou" \
        "t-len",
        type=int,
        default=1000,
        help="length of training rollouts AND length at which env is reset",
    )
    parser.add_argument(
        "--episode-len",
        type=int,
        default=1e8,
        help="Number of environment episodes",
    )
    parser.add_argument(
        "--use-reputation",
        type=bool,
        default=False,
        help="Give agent other agents' reputation",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log",
    )
    args = parser.parse_args()
    return args

def main(args):
    # Config
    network_mode = False
    env_name = args.env_name
    num_agents = args.num_agents
    rollout_len = args.rollout_len
    total_timesteps = args.total_episodes
    use_reputation = args.use_reputation    
    log_file = args.log_file
    alpha = 1
    beam_penalty = 0
    broadcast_mode = 2
    broadcast_radius =9
    env = create_env(env=env_name, num_agents=num_agents,
                     use_collective_reward=False, inequity_averse_reward=False, 
                     alpha=alpha, beta=0, use_reputation=use_reputation, 
                     beam_penalty=beam_penalty, broadcast_mode = broadcast_mode,
                     broadcast_radius=broadcast_radius,network_mode=network_mode)
    lr = 0.0001
    batch_size = 100
    gamma = 0.99
    epsilon = [1, 0.1, 1000000]
    grad_clip = 1
    update_frequency = 10
    replay_capacity = 2000
    conv_filter = [ [32, [3, 3], 1],[16,[3,3],1] ]
    fc_hidden = [32,16]
    tau = 0.005
    num_episodes = 10000
    episode_length = rollout_len
    algo = mult_agent_dqn(device=device,env = env, epsilon_decay=epsilon, 
                          update_freq=update_frequency, conv_filters=conv_filter,
                          fc_filters=fc_hidden,lr=lr, replay_capacity=replay_capacity, 
                          batch_size=batch_size,gamma=gamma,grad_clip=grad_clip, 
                          num_episode=num_episodes,tau=tau,episode_len=episode_length,
                          use_cnn=True, log_file=log_file)
    algo.train()

if __name__ == "__main__":
    args = parse_args()
    main(args)
