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

#from agent.TD3 import PIRLAgent, AgentConfig, train
from agent.TD3_PIRL_ray import PIRLAgent, AgentConfig, train_distributed


@dataclass
class CaseConfig:
    env_module: str
    log_dir: str
    num_episodes: int
    checkpoint_freq: int
    critic_hidden_dims: tuple
    critic_lr: float
    actor_lr: float    
    num_collocations: tuple
    learn_policy_noise: float
    learn_noise_clip: float
    policy_update_freq: int
    initial_exploration_num: int
    exploration_noise: float
    weight_schedule: dict|None = None

CASE_CONFIGS = {
    "1D": CaseConfig(
        # Basic settings
        env_module   = "examples.env_1d_reach",
        log_dir      = "logs/1D",
        num_episodes = 100_000,
        checkpoint_freq  = 10_000,
        # NN settings
        critic_hidden_dims = (32,32,32,32),
        critic_lr = 1e-4,
        actor_lr  = 1e-4,
        weight_schedule = {
            "initial": (0.5, 0.5, 0.5), 
            "final":   (0.0, 1.0, 1.0),
            "center": 25_000,
            "sharpness": 0.0005 }, 
        # PINN
        num_collocations   = (1000, 100, 100),
        learn_policy_noise = 0.3, #0.2
        learn_noise_clip   = 1.0, #0.5
        # RL (policy)
        policy_update_freq = 5,
        initial_exploration_num = 1000,
        exploration_noise = 0.2,
    ),
    "2D": CaseConfig(
        env_module   = "examples.env_2d_avoid",
        log_dir      = "logs/2D",
        num_episodes = 20000,
        checkpoint_freq = 2000,
        # NN settings
        critic_hidden_dims = (64,64),
        critic_lr = 1e-4,
        actor_lr  = 1e-4,
        # PINN
        num_collocations = (1000, 100, 100),
        learn_policy_noise = 0.2,
        learn_noise_clip   = 0.5,
        # RL
        policy_update_freq=50,
        initial_exploration_num=1000,
        exploration_noise=0.2,
    ),
    "drift": CaseConfig(
        env_module   ="examples.env_drifting_control",
        log_dir      ="logs/drift",
        num_episodes = 30000,
        checkpoint_freq  = 1000,
        # NN settings
        critic_hidden_dims = (64,64),
        critic_lr = 1e-4,
        actor_lr  = 1e-4,
        # PINN        
        num_collocations = (2000, 200, 200),
        learn_policy_noise = 0.2,
        learn_noise_clip   = 0.5,
        # RL
        policy_update_freq=50,
        initial_exploration_num=2000,
        exploration_noise=0.2,
    ),
}

def get_loss_setting(method, case):
    if method == "td3":
        return [1.0, 0.0, 0.0], None

    if method == "pinn":
        return [0.0, 1.0, 1.0], None

    if method == "fixed":
        return [1.0, 1.0, 1.0], None

    if method == "scheduling":
        cfg = CASE_CONFIGS[case]
        loss_weights = cfg.weight_schedule['initial']
        weight_schedule = cfg.weight_schedule
        return loss_weights, weight_schedule

    raise ValueError(f"Unknown method: {method}")

def make_env(case):
    cfg = CASE_CONFIGS[case]
    module = importlib.import_module(cfg.env_module)
    return module.Env()

def make_env_cls(case):
    cfg = CASE_CONFIGS[case]
    module = importlib.import_module(cfg.env_module)
    return module.Env

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
    parser.add_argument("--num_workers", default=4, type=int)
    args = parser.parse_args()

    set_seed(args.seed)

    case_cfg = CASE_CONFIGS[args.case]
    env_cls = make_env_cls(args.case)
    env = make_env(args.case)
    
    agent_config = AgentConfig(
        state_dim  = env.state_dim,
        action_dim = env.action_dim,
        critic_hidden_dims= case_cfg.critic_hidden_dims,
        critic_lr  = case_cfg.critic_lr,
        actor_lr   = case_cfg.actor_lr,
        learn_policy_noise = case_cfg.learn_policy_noise,
        learn_noise_clip   = case_cfg.learn_noise_clip,
    )

    if args.checkpoint is None:
        agent = PIRLAgent(agent_config, device=args.device)
    else:
        agent = PIRLAgent.from_checkpoint(args.checkpoint)

    loss_weights, weight_schedule = get_loss_setting(args.method, args.case)

    print("--------------------------------------------")
    print(f"Case   : {args.case}")
    print(f"Method : {args.method}")
    print(f"Seed   : {args.seed}")
    print(f"Device : {args.device}")
    print(f"Log dir: {case_cfg.log_dir}")
    print("--------------------------------------------")

    # train(
    #     env,
    #     agent,
    #     num_episodes = case_cfg.num_episodes,
    #     seed         = args.seed,
    #     log_dir      = case_cfg.log_dir,
    #     checkpoint_freq = case_cfg.checkpoint_freq,
    #     verbose         = args.verbose,
    #     loss_weights    = loss_weights,
    #     weight_schedule = weight_schedule,
    #     num_collocations= case_cfg.num_collocations,
    #     policy_update_freq= case_cfg.policy_update_freq,
    #     initial_exploration_num = case_cfg.initial_exploration_num,
    #     exploration_noise = case_cfg.exploration_noise,
    # )

    train_distributed(
        env_cls=env_cls,
        agent=agent,
        num_iterations=case_cfg.num_episodes,
        seed=args.seed,
        num_workers=args.num_workers,
        log_dir=case_cfg.log_dir,
        checkpoint_freq=case_cfg.checkpoint_freq,
        verbose=args.verbose,
        device=args.device,
        loss_weights=loss_weights,
        weight_schedule=weight_schedule,
        num_collocations=case_cfg.num_collocations,
        initial_exploration_num=case_cfg.initial_exploration_num,
        exploration_noise=case_cfg.exploration_noise,
        minibatch_size=128,
        policy_update_freq=case_cfg.policy_update_freq,
        target_update_rate=0.005,
    )

if __name__ == "__main__":
    main()

