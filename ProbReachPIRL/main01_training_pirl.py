"""
  Training of PIRL agent for planer system
"""
import numpy as np
import random
import argparse
import torch

import sys, os
sys.path.append(os.pardir)

from agent.TD3 import PIRLAgent, AgentConfig
from agent.TD3 import train
from examples.planer_env import Env

parser = argparse.ArgumentParser()
parser.add_argument("--method", default="scheduled")     # "td3", "pinn", "scheduled"
parser.add_argument("--seed",   default=1, type=int)  
args = parser.parse_args()

# Set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Environment
env   = Env()

agent_config = AgentConfig(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        )

agent = PIRLAgent(agent_config, device="cpu")

print("--------------------------------------------")
print(f"Method: {args.method}, Seed: {args.seed}")
print("--------------------------------------------")

LOG_DIR = 'logs'

if args.method == "td3": 
    loss_weights = [1, 0, 0] # "td3", "hbj", "bdr"
    weight_schedule = None
    
elif args.method == "pinn":
    loss_weights = [0, 1, 1] # "td3", "hbj", "bdr"
    weight_schedule = None

elif args.method == "scheduled":
    loss_weights = [1, 1e-2, 1e-1]
    weight_schedule = (3000, 0.005, 0, 1) # (ep_Half, steepness, del_start, del_end)     

train(env, 
      agent, 
      num_episodes = 10000, 
      seed      = 1,
      log_dir   = LOG_DIR,
      checkpoint_freq = 500,
      verbose   = 1,
      loss_weights = loss_weights,
      weight_schedule = weight_schedule,
      num_collocations = (64, 32, 32)
      )


 