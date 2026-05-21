# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from agent.TD3 import PIRLAgent

def plot_value(ax, agent, T, x, label):       
    T_arr = np.full_like(x, T)
    states = np.stack([T_arr, x], axis=-1).reshape(-1, 2)
    value = agent.get_value(states)    
    ax.plot(x, value, label=label)

def true_reach_probability(T, x, mu=1.0, sigma=1.0, x_target=2.0):
    """
    Finite-horizon reach probability:
        P_x( sup_{0<=s<=T} X_s >= x_target )
    for dX = mu dt + sigma dW
    """
    x = np.asarray(x, dtype=np.float64)

    if T <= 0:
        return (x >= x_target).astype(np.float64)

    prob = np.ones_like(x, dtype=np.float64)
    mask = x < x_target

    z1 = (x[mask] - x_target + mu * T) / (sigma * np.sqrt(T))
    z2 = (x[mask] - x_target - mu * T) / (sigma * np.sqrt(T))

    prob[mask] = norm.cdf(z1) + np.exp(2.0 * mu * (x_target - x[mask]) / (sigma**2)) * norm.cdf(z2)
    prob = np.clip(prob, 0.0, 1.0)

    return prob


#############################
# Settings
#############################
T = 2.0
num_grid = 100
x = np.linspace(-2.0, 2.0, num_grid)

agents = {
    "RL":   PIRLAgent.from_checkpoint('logs/1D/0508_1853_seed_1/ckpt-20000'),
    "PINN": PIRLAgent.from_checkpoint('logs/1D/0508_0739_seed_1/ckpt-20000'),
    "PIRL (Proposed)": PIRLAgent.from_checkpoint('logs/1D/0509_0555_seed_1/ckpt-20000'),
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

#############################
# Figure
#############################
fig, ax = plt.subplots()

value_true = true_reach_probability(T, x, mu=1.0, sigma=1.0, x_target=2.0)
ax.plot(x, value_true, "k--", linewidth=2.5,label="Ground truth")

for tag, agent in agents.items():   
    plot_value(ax, agent, T, x, tag)

ax.set_xlabel(r"$x$")
ax.set_ylabel("Reach Probability")
ax.set_xlim([-2.0, 2.0])
ax.set_ylim([0, 1.0])
ax.legend( loc="upper left", frameon=True)

plt.savefig(
    "plot/fig_1d_value_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

#fig.tight_layout()