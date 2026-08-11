# -*- coding: utf-8 -*-
"""Stratify drift value-calibration error by distance from TD3 occupancy.

The checkpoint format does not retain replay memory.  This script therefore
reconstructs a reference occupancy distribution by rolling out the TD3 policy
from the training reset distribution.  A common set of perturbed states is
ranked by k-nearest-neighbour distance to that occupancy set.  Each model's
learned value is then compared with Monte Carlo reachability under its own
policy in the same distance bins.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from examples.env_drifting_control import Env
# The network/checkpoint schema is shared with TD3_PIRL_ray, while this module
# avoids importing Ray for a read-only evaluation job.
from agent.TD3 import AgentConfig, PIRLAgent
from plot_drift_mc_reachability import rollout_reach_probability


DEFAULT_MODELS = {
    "TD3 random tau 10M": "logs/drift/td3_T01_randT/up10M_scale10_mix334_randT/ckpt-10000000",
    "PIRL 10M": (
        "scheduling_experiment/fixed2randT_replayHJB_7Mto10M_restart2/round_002/"
        "hold0015_R3/train/0701_0606_hold0015_R3_seed_1/ckpt-10000000"
    ),
    "PIRL mixed 15M": (
        "scheduling_experiment/fixed2randT_mixedHJB_10Mto15M/round_004/"
        "mix85_back003_R5/train/0708_2216_mix85_back003_R5_seed_1/ckpt-15000000"
    ),
    "TD3 fixed tau 10M": "logs/drift/td3_T01/up10M_scale10_mix334/ckpt-10000000",
}


def load_agent(path: str, device: str) -> PIRLAgent:
    """Load ray-era checkpoints with the compatible non-Ray network class."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=device)
    allowed = {field.name for field in fields(AgentConfig)}
    config = AgentConfig(**{k: v for k, v in checkpoint["config"].items() if k in allowed})
    agent = PIRLAgent(config=config, device=device)
    agent.itr = checkpoint.get("itr", 0)
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])
    return agent


def configure_training_env() -> Env:
    os.environ["DRIFT_DT"] = "0.1"
    os.environ["DRIFT_RESET_SCALE"] = "1.0"
    os.environ["DRIFT_RESET_MODE"] = "mixture"
    os.environ["DRIFT_RESET_MIXTURE_PROBS"] = "0.3,0.3,0.4"
    os.environ["DRIFT_RESET_T_MODE"] = "random"
    os.environ["DRIFT_RESET_T_MIN"] = "0.0"
    os.environ["DRIFT_RESET_T_MAX"] = "5.0"
    return Env()


def rollout_occupancy(agent, env: Env, episodes: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    states = []
    for _ in range(episodes):
        scaled = env.reset()
        while True:
            states.append(np.asarray(scaled, dtype=np.float32).copy())
            action = agent.get_action(np.asarray(scaled)[None, :])[0]
            scaled, _, done, _ = env.step(action)
            if done:
                break
    return np.asarray(states, dtype=np.float32)


def make_eval_states(
    occupancy_scaled: np.ndarray,
    n_states: int,
    max_radius: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Perturb occupancy anchors with radii spanning zero to max_radius."""
    rng = np.random.default_rng(seed)
    replace = len(occupancy_scaled) < n_states
    anchors = occupancy_scaled[rng.choice(len(occupancy_scaled), n_states, replace=replace)]
    directions = rng.normal(size=anchors.shape).astype(np.float32)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    radii = rng.uniform(0.0, max_radius, size=(n_states, 1)).astype(np.float32)
    candidate = np.clip(anchors + radii * directions, -1.0, 1.0)
    return candidate, radii[:, 0]


def metrics(error: np.ndarray) -> dict[str, float]:
    over = np.maximum(error, 0.0)   # V > MC: optimistic / risk underestimation
    under = np.maximum(-error, 0.0) # V < MC: conservative
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(error ** 2)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias_mc_minus_v": float(np.mean(-error)),
        "optimistic_mae": float(np.mean(over)),
        "optimistic_mse": float(np.mean(over ** 2)),
        "conservative_mae": float(np.mean(under)),
        "conservative_mse": float(np.mean(under ** 2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="plot/drift_error_validation")
    parser.add_argument("--occupancy_episodes", type=int, default=1024)
    parser.add_argument("--num_states", type=int, default=512)
    parser.add_argument("--num_rollouts", type=int, default=256)
    parser.add_argument("--max_radius", type=float, default=0.40)
    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("--num_bins", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    env = configure_training_env()
    models = {
        name: load_agent(path, args.device)
        for name, path in DEFAULT_MODELS.items()
    }

    td3 = models["TD3 random tau 10M"]
    occupancy = rollout_occupancy(td3, env, args.occupancy_episodes, args.seed)
    eval_scaled, generated_radius = make_eval_states(
        occupancy, args.num_states, args.max_radius, args.seed + 1
    )
    eval_phys = env.unscale_state(eval_scaled)

    # Remove states that became terminal after clipping/perturbation: their
    # exact boundary values would otherwise dominate the OOD comparison.
    valid = ~(env.is_target(eval_phys) | env.is_unsafe(eval_phys) | (eval_phys[:, 0] < env.dt))
    eval_scaled = eval_scaled[valid]
    eval_phys = eval_phys[valid]
    generated_radius = generated_radius[valid]

    tree = cKDTree(occupancy)
    distances, _ = tree.query(eval_scaled, k=min(args.knn, len(occupancy)))
    if distances.ndim == 1:
        distances = distances[:, None]
    occupancy_distance = distances.mean(axis=1)

    quantiles = np.linspace(0.0, 1.0, args.num_bins + 1)
    edges = np.quantile(occupancy_distance, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_index = np.clip(np.digitize(occupancy_distance, edges[1:-1]), 0, args.num_bins - 1)

    rows = []
    raw = {
        "states_scaled": eval_scaled,
        "states_phys": eval_phys,
        "generated_radius": generated_radius,
        "occupancy_distance": occupancy_distance,
        "distance_bin": bin_index,
        "occupancy_scaled": occupancy,
    }
    for model_i, (name, agent) in enumerate(models.items()):
        value = np.clip(agent.get_value(eval_scaled).reshape(-1), 0.0, 1.0)
        # The rollout helper expects a square grid only for reshaping.  Pad to
        # the next square and discard padding afterward.
        side = int(np.ceil(np.sqrt(len(eval_phys))))
        padded_n = side * side
        pad = padded_n - len(eval_phys)
        phys_pad = np.concatenate([eval_phys, np.repeat(eval_phys[-1:], pad, axis=0)], axis=0)
        prob, _, _ = rollout_reach_probability(
            agent, env, phys_pad, side, args.num_rollouts, 8192, False,
            args.seed + 1000 * (model_i + 1),
        )
        prob = prob.reshape(-1)[:len(eval_phys)]
        error = value - prob
        raw[f"{name}__value"] = value
        raw[f"{name}__mc"] = prob
        raw[f"{name}__error"] = error

        for b in range(args.num_bins):
            mask = bin_index == b
            row = {
                "method": name,
                "distance_bin": b,
                "n": int(mask.sum()),
                "distance_mean": float(occupancy_distance[mask].mean()),
                "distance_min": float(occupancy_distance[mask].min()),
                "distance_max": float(occupancy_distance[mask].max()),
                "mean_mc": float(prob[mask].mean()),
                "mean_v": float(value[mask].mean()),
            }
            row.update(metrics(error[mask]))
            rows.append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ood_value_accuracy_by_distance.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out_dir / "ood_value_accuracy_raw.npz", **raw)

    print(f"occupancy states: {len(occupancy)}")
    print(f"valid evaluation states: {len(eval_phys)}")
    print(f"saved: {csv_path}")
    for row in rows:
        print(
            f"{row['method']:20s} bin={row['distance_bin']} n={row['n']:3d} "
            f"d={row['distance_mean']:.4f} MAE={row['mae']:.4f} "
            f"MSE={row['mse']:.4f} optMSE={row['optimistic_mse']:.4f} "
            f"consMSE={row['conservative_mse']:.4f}"
        )


if __name__ == "__main__":
    main()
