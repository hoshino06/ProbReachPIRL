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
import os
import sys
from dataclasses import dataclass

#from agent.TD3 import PIRLAgent, AgentConfig, train
from agent.TD3_PIRL_ray import PIRLAgent, AgentConfig, train_distributed


parser = argparse.ArgumentParser()
parser.add_argument("--case",   default="drift")       # "1D", "2D", "drift"
parser.add_argument("--method", default="scheduling") # "td3", "pinn", "scheduling"
parser.add_argument("--seed",   default=1, type=int)  
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--device", default="auto")  # auto, cpu, cuda
parser.add_argument("--verbose", default=1, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--num_updates", default=1_000_000, type=int,
                    help="Number of training updates.")
parser.add_argument("--num_collocations", default=None, type=int, nargs=3,
                    metavar=("NPDE", "NTARGET", "NAVOID"),
                    help="Override PINN collocation counts.")
parser.add_argument("--pinn_sample_mode", default="uniform", choices=("uniform", "replay"),
                    help="Sample HJB PDE points uniformly or from replay memory.")
parser.add_argument("--pinn_replay_fraction", default=1.0, type=float,
                    help="Fraction of HJB PDE points drawn from replay memory when pinn_sample_mode=replay.")
parser.add_argument("--pinn_replay_jitter", default=0.0, type=float,
                    help="Gaussian jitter in scaled state units for replay HJB PDE points.")
parser.add_argument("--learner_num_gpus", default=None, type=float,
                    help="Ray GPU resource assigned to each Learner actor. Omit for legacy behavior.")
parser.add_argument("--hjb_laplacian_mode", default="loop", choices=("loop", "batched"),
                    help="How to compute diagonal Hessian terms in the HJB diffusion term.")
parser.add_argument("--schedule_center", default=None, type=int,
                    help="Override weight_schedule center for scheduling method.")
parser.add_argument("--schedule_sharpness", default=None, type=float,
                    help="Override weight_schedule sharpness for scheduling method.")
parser.add_argument("--schedule_final", default=None, type=float, nargs=3,
                    metavar=("WTD3", "WHJB", "WBDR"),
                    help="Override final weights for scheduling method.")
parser.add_argument("--schedule_initial", default=None, type=float, nargs=3,
                    metavar=("WTD3", "WHJB", "WBDR"),
                    help="Override initial weights for scheduling method.")
parser.add_argument("--log_tag", default=None,
                    help="Optional tag appended to the TensorBoard run directory.")
parser.add_argument("--drift_reset_scale", default=1.0, type=float,
                    help="Multiplier for the drift environment reset distribution.")
parser.add_argument("--drift_reset_mode", default="full", choices=("full", "mixture"),
                    help="Drift reset distribution mode.")
parser.add_argument("--drift_reset_mixture_probs", default="0.45,0.45,0.10",
                    help="Comma-separated beta_r,ey_epsi,full probabilities for mixture reset.")
parser.add_argument("--drift_reset_t_mode", default="fixed", choices=("fixed", "random"),
                    help="Use Tmax or a random remaining time at drift reset.")
parser.add_argument("--drift_reset_t_min", default=0.0, type=float,
                    help="Lower bound for random drift reset remaining time.")
parser.add_argument("--drift_reset_t_max", default=None, type=float,
                    help="Maximum remaining time for random drift reset. Defaults to env.Tmax.")
parser.add_argument("--drift_dt", default=None, type=float,
                    help="Override the drifting-control environment time step.")
parser.add_argument("--initial_exploration_policy", default="random",
                    choices=("random", "policy"),
                    help="Use random actions or the current policy to fill the initial replay buffer.")
parser.add_argument("--schedule_time_base", default="global", choices=("global", "local"),
                    help="Use absolute update count or fine-tune-local count for weight scheduling.")
parser.add_argument("--log_dir_override", default=None,
                    help="Override CaseConfig.log_dir for this run.")
parser.add_argument("--replay_memory_size", default=None, type=int,
                    help="Override replay memory size.")
parser.add_argument("--exploration_noise", default=None, type=float,
                    help="Override rollout exploration noise.")
parser.add_argument("--policy_update_freq", default=None, type=int,
                    help="Override delayed policy update frequency.")
parser.add_argument("--initial_exploration_num", default=None, type=int,
                    help="Override initial replay buffer fill size.")
parser.add_argument("--critic_lr", default=None, type=float,
                    help="Override critic learning rate.")
parser.add_argument("--actor_lr", default=None, type=float,
                    help="Override actor learning rate.")
parser.add_argument("--learning_rate", default=None, type=float,
                    help="Override both actor and critic learning rates.")
args = parser.parse_args()


@dataclass
class CaseConfig:
    env_module: str
    log_dir: str
    checkpoint_freq: int
    critic_hidden_dims: tuple
    critic_lr: float
    actor_lr: float    
    num_collocations: tuple
    learn_policy_noise: float
    learn_noise_clip: float
    policy_update_freq: int
    replay_memory_size: int
    initial_exploration_num: int
    exploration_noise: float
    minibatch_size: int
    weight_schedule: dict|None = None

CASE_CONFIGS = {
    "1D": CaseConfig(
        # Basic settings
        env_module   = "examples.env_1d_reach",
        log_dir      = f"logs/1D/{args.method}",
        checkpoint_freq  = 50_000,
        # NN settings
        critic_hidden_dims = (32,32,32),
        critic_lr = 1e-4,
        actor_lr  = 1e-4,
        weight_schedule = {
            "initial": (1.0, 0.0, 0.0), #(TD3, HJB, BDR)
            "final":   (0.0, 1.0, 1.0), 
            "center": 50_000,
            "sharpness": 0.0001 }, 
        # PINN
        num_collocations   = (1000, 100, 100),
        learn_policy_noise = 0.3, #0.2
        learn_noise_clip   = 1.0, #0.5
        # RL (policy)
        policy_update_freq = 10,
        replay_memory_size = 5000,
        initial_exploration_num = 1000,
        exploration_noise = 0.2,
        minibatch_size=128
    ),
    "2D": CaseConfig(
        env_module   = "examples.env_2d_avoid",
        log_dir      = "logs/2D",
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
        policy_update_freq = 10,
        replay_memory_size = 5000,
        initial_exploration_num=1000,
        exploration_noise=0.2,
        minibatch_size=128
    ),
    "drift": CaseConfig(
        env_module   ="examples.env_drifting_control",
        log_dir      =f"logs/drift/{args.method}",
        checkpoint_freq  = 100_000,
        # NN settings
        critic_hidden_dims = (64,64,64),
        critic_lr = 1e-4,
        actor_lr  = 1e-4,
        weight_schedule = {
            "initial": (1.0, 0.0, 0.0),  # (TD3, HJB, BDR)
            "final":   (0.0, 1.0, 1.0),
            "center": 500_000,
            "sharpness": 1.0e-5,
        },
        # PINN
        num_collocations = (1000, 100, 100),
        learn_policy_noise = 0.2,
        learn_noise_clip   = 0.5,
        # RL
        policy_update_freq = 10,
        replay_memory_size = 100_000,
        initial_exploration_num = 10_000,
        exploration_noise=0.2,
        minibatch_size = 256
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
        if cfg.weight_schedule is None:
            raise ValueError(f"Case {case} does not define a weight schedule.")
        schedule = {
            "initial": tuple(cfg.weight_schedule["initial"]),
            "final": tuple(cfg.weight_schedule["final"]),
            "center": cfg.weight_schedule["center"],
            "sharpness": cfg.weight_schedule["sharpness"],
        }
        if args.schedule_center is not None:
            schedule["center"] = args.schedule_center
        if args.schedule_sharpness is not None:
            schedule["sharpness"] = args.schedule_sharpness
        if args.schedule_initial is not None:
            schedule["initial"] = tuple(args.schedule_initial)
        if args.schedule_final is not None:
            schedule["final"] = tuple(args.schedule_final)
        loss_weights = schedule["initial"]
        return loss_weights, schedule

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

def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def main():

    print(f"[seed={args.seed}] started")
    set_seed(args.seed)
    os.environ["DRIFT_RESET_SCALE"] = str(args.drift_reset_scale)
    os.environ["DRIFT_RESET_MODE"] = str(args.drift_reset_mode)
    os.environ["DRIFT_RESET_MIXTURE_PROBS"] = str(args.drift_reset_mixture_probs)
    os.environ["DRIFT_RESET_T_MODE"] = str(args.drift_reset_t_mode)
    os.environ["DRIFT_RESET_T_MIN"] = str(args.drift_reset_t_min)
    if args.drift_reset_t_max is not None:
        os.environ["DRIFT_RESET_T_MAX"] = str(args.drift_reset_t_max)
    if args.drift_dt is not None:
        os.environ["DRIFT_DT"] = str(args.drift_dt)

    case_cfg = CASE_CONFIGS[args.case]
    if args.log_dir_override is not None:
        case_cfg.log_dir = args.log_dir_override
    if args.replay_memory_size is not None:
        case_cfg.replay_memory_size = args.replay_memory_size
    if args.exploration_noise is not None:
        case_cfg.exploration_noise = args.exploration_noise
    if args.policy_update_freq is not None:
        case_cfg.policy_update_freq = args.policy_update_freq
    if args.initial_exploration_num is not None:
        case_cfg.initial_exploration_num = args.initial_exploration_num
    if args.learning_rate is not None:
        case_cfg.critic_lr = args.learning_rate
        case_cfg.actor_lr = args.learning_rate
    if args.critic_lr is not None:
        case_cfg.critic_lr = args.critic_lr
    if args.actor_lr is not None:
        case_cfg.actor_lr = args.actor_lr
    env_cls = make_env_cls(args.case)
    env = make_env(args.case)
    
    agent_config = AgentConfig(
        state_dim  = env.state_dim,
        action_dim = env.action_dim,
        critic_hidden_dims= case_cfg.critic_hidden_dims,
        critic_lr  = case_cfg.critic_lr,
        actor_lr   = case_cfg.actor_lr,
        replay_memory_size = case_cfg.replay_memory_size,
        learn_policy_noise = case_cfg.learn_policy_noise,
        learn_noise_clip   = case_cfg.learn_noise_clip,
        hjb_laplacian_mode = args.hjb_laplacian_mode,
    )

    if args.checkpoint is None:
        agent = PIRLAgent(agent_config, device=args.device)
    else:
        agent = PIRLAgent.from_checkpoint(args.checkpoint)
        agent.config.replay_memory_size = case_cfg.replay_memory_size
        agent.config.critic_lr = case_cfg.critic_lr
        agent.config.actor_lr = case_cfg.actor_lr
        agent.config.hjb_laplacian_mode = args.hjb_laplacian_mode
        set_optimizer_lr(agent.critic_optimizer, case_cfg.critic_lr)
        set_optimizer_lr(agent.actor_optimizer, case_cfg.actor_lr)

    loss_weights, weight_schedule = get_loss_setting(args.method, args.case)

    print("--------------------------------------------")
    print(f"Case   : {args.case}")
    print(f"Method : {args.method}")
    print(f"Seed   : {args.seed}")
    print(f"Device : {args.device}")
    print(f"Log dir: {case_cfg.log_dir}")
    print(f"Updates: {args.num_updates}")
    print(f"Colloc.: {tuple(args.num_collocations) if args.num_collocations is not None else case_cfg.num_collocations}")
    print(f"PINN sample mode: {args.pinn_sample_mode}")
    if args.pinn_sample_mode == "replay":
        print(f"PINN replay fraction: {args.pinn_replay_fraction}")
        print(f"PINN replay jitter: {args.pinn_replay_jitter}")
    print(f"Learner GPUs: {args.learner_num_gpus if args.learner_num_gpus is not None else 'legacy'}")
    print(f"HJB Lap.: {args.hjb_laplacian_mode}")
    print(f"Reset scale: {args.drift_reset_scale}")
    print(f"Reset mode: {args.drift_reset_mode}")
    print(f"Reset T mode: {args.drift_reset_t_mode}")
    if args.drift_reset_mode == "mixture":
        print(f"Reset mixture probs: {args.drift_reset_mixture_probs}")
    if args.drift_reset_t_mode == "random":
        print(f"Reset T range: [{args.drift_reset_t_min}, {args.drift_reset_t_max if args.drift_reset_t_max is not None else env.Tmax}]")
    print(f"Drift dt: {env.dt if args.case == 'drift' else '-'}")
    print(f"Replay memory size: {case_cfg.replay_memory_size}")
    print(f"Critic LR: {case_cfg.critic_lr}")
    print(f"Actor LR: {case_cfg.actor_lr}")
    print(f"Exploration noise: {case_cfg.exploration_noise}")
    print(f"Policy update freq: {case_cfg.policy_update_freq}")
    print(f"Initial exploration num: {case_cfg.initial_exploration_num}")
    print(f"Initial exploration: {args.initial_exploration_policy}")
    if weight_schedule is not None:
        print(f"Schedule: center={weight_schedule['center']}, sharpness={weight_schedule['sharpness']}, final={weight_schedule['final']}")
        print(f"Schedule time base: {args.schedule_time_base}")
    print(f"Log tag: {args.log_tag if args.log_tag else '-'}")
    print("--------------------------------------------")
    sys.stdout.flush()

    # train(
    #     env,
    #     agent,
    #     num_iterations = args.num_updates,
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
        num_iterations=args.num_updates,
        seed=args.seed,
        num_workers=args.num_workers,
        log_dir=case_cfg.log_dir,
        checkpoint_freq=case_cfg.checkpoint_freq,
        verbose=args.verbose,
        device=args.device,
        learner_num_gpus=args.learner_num_gpus,
        log_tag=args.log_tag,
        loss_weights=loss_weights,
        weight_schedule=weight_schedule,
        weight_schedule_time_base=args.schedule_time_base,
        num_collocations=tuple(args.num_collocations) if args.num_collocations is not None else case_cfg.num_collocations,
        pinn_sample_mode=args.pinn_sample_mode,
        pinn_replay_fraction=args.pinn_replay_fraction,
        pinn_replay_jitter=args.pinn_replay_jitter,
        initial_exploration_num=case_cfg.initial_exploration_num,
        initial_exploration_policy=args.initial_exploration_policy,
        exploration_noise=case_cfg.exploration_noise,
        minibatch_size=case_cfg.minibatch_size,
        policy_update_freq=case_cfg.policy_update_freq,
        target_update_rate=0.005,
    )

    print(f"[seed={args.seed}] finished")
    
if __name__ == "__main__":
    main()
