# main01_training_pirl.py
"""
Main training script for PIRL agent.

Examples:
    python train_main.py --case 1d --method td3 --seed 1
    python train_main.py --case 2d --method scheduling --seed 1
    python train_main.py --case drift --method scheduling --seed 1
"""

import numpy as np
import random
import argparse
import torch
import importlib
from dataclasses import dataclass

from agent.TD3 import PIRLAgent, AgentConfig, train

@dataclass
class CaseConfig:
    env_module: str
    log_dir: str
    num_episodes: int
    num_collocations: tuple
    checkpoint_freq: int
    policy_update_freq: int
    initial_exploration_num: int
    exploration_noise: float
    critic_lr: float = 1e-4
    actor_lr: float = 1e-4
    learn_policy_noise: float = 0.5
    learn_noise_clip: float = 0.5

CASE_CONFIGS = {
    "1d": CaseConfig(
        env_module   = "examples.env_1d_reach",
        log_dir      = "logs/1D",
        num_episodes = 20000,
        num_collocations = (1000, 100, 100),
        checkpoint_freq=500,
        policy_update_freq=50,
        initial_exploration_num=1000,
        exploration_noise=0.2,
    ),
    "2d": CaseConfig(
        env_module   = "examples.env_2d_avoid",
        log_dir      = "2D",
        num_episodes = 20000,
        num_collocations = (1000, 100, 100),
        checkpoint_freq=500,
        policy_update_freq=50,
        initial_exploration_num=1000,
        exploration_noise=0.2,
    ),
    "drift": CaseConfig(
        env_module   ="examples.env_drifting_control",
        log_dir      ="logs/drift",
        num_episodes = 30000,
        num_collocations = (2000, 200, 200),
        checkpoint_freq  = 1000,
        policy_update_freq=50,
        initial_exploration_num=2000,
        exploration_noise=0.2,
        critic_lr=1e-4,
        actor_lr=1e-4,
    ),
}

def get_loss_setting(method):
    if method == "td3":
        return [1.0, 0.0, 0.0], None

    if method == "pinn":
        return [0.0, 1.0, 1.0], None

    if method == "fixed":
        return [1.0, 1.0, 1.0], None

    if method == "scheduling":
        loss_weights = [1.0, 1.0, 1.0]
        weight_schedule = (10000, 5e-4, 1e-2, 1.0)
        return loss_weights, weight_schedule

    raise ValueError(f"Unknown method: {method}")

def make_env(case):
    cfg = CASE_CONFIGS[case]
    module = importlib.import_module(cfg.env_module)
    return module.Env()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--case",   default="drift")       # "1D", "2D", "drift"
    parser.add_argument("--method", default="scheduling") # "td3", "pinn", "scheduling"
    parser.add_argument("--seed",   default=1, type=int)  
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")  # auto, cpu, cuda
    parser.add_argument("--verbose", default=1, type=int)
    args = parser.parse_args()

    set_seed(args.seed)

    case_cfg = CASE_CONFIGS[args.case]
    env = make_env(args.case)
    
    agent_config = AgentConfig(
        state_dim  = env.state_dim,
        action_dim = env.action_dim,
        critic_lr  = case_cfg.critic_lr,
        actor_lr   = case_cfg.actor_lr,
        learn_policy_noise = case_cfg.learn_policy_noise,
        learn_noise_clip   = case_cfg.learn_noise_clip,
    )

    if args.checkpoint is None:
        agent = PIRLAgent(agent_config, device=args.device)
    else:
        agent = PIRLAgent.from_checkpoint(args.checkpoint)

    loss_weights, weight_schedule = get_loss_setting(args.method)

    print("--------------------------------------------")
    print(f"Case   : {args.case}")
    print(f"Method : {args.method}")
    print(f"Seed   : {args.seed}")
    print(f"Device : {args.device}")
    print(f"Log dir: {case_cfg.log_dir}")
    print("--------------------------------------------")

    train(
        env,
        agent,
        num_episodes = case_cfg.num_episodes,
        seed         = args.seed,
        log_dir      = case_cfg.log_dir,
        checkpoint_freq = case_cfg.checkpoint_freq,
        verbose         = args.verbose,
        loss_weights    = loss_weights,
        weight_schedule = weight_schedule,
        num_collocations= case_cfg.num_collocations,
        policy_update_freq= case_cfg.policy_update_freq,
        initial_exploration_num = case_cfg.initial_exploration_num,
        exploration_noise = case_cfg.exploration_noise,
    )
    

if __name__ == "__main__":
    main()

