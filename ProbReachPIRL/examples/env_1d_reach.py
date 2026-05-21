# -*- coding: utf-8 -*-
"""
1D reachability environment for RL / PINN / PIRL
State: [remaining_time, x]
Target: x >= x_target
Dynamics: dX = u dt + sigma dW,   u in [-u_max, u_max]
"""

import numpy as np
from scipy.stats import qmc, norm
import matplotlib.pyplot as plt

class Env(object):
    
    # problem setting
    Tmax = 2.0
    x_min = -2.0
    x_max = 2.0
    x_target = 2.0
    sigma_val = 1.0
    u_max = 1.0
    
    def __init__(self):
        self.dt = 0.1
        self.state_dim = 2     # [T, x]
        self.action_dim = 1

    ################################################
    # Common dynamics model
    ################################################
    def drift_and_diffusion(self, X, U):
        """
        Continuous-time dynamics:
            dY = f(Y, U) dt + sigma(Y,U) dW
        where Y = [T, x]

        X: shape (2,) or (N,2)
        U: shape (1,) or (N,1)
        """
        X = np.asarray(X, dtype=np.float32)
        U = np.asarray(U, dtype=np.float32)

        single = (X.ndim == 1)
        if single:
            X = X[None, :]
            U = U[None, :]

        T = X[:, 0]
        x = X[:, 1]
        u = U[:, 0]

        # clip action
        u = np.clip(u, -self.u_max, self.u_max)

        ######################
        # Drift
        ######################
        f = np.zeros_like(X, dtype=np.float32)
        f[:, 0] = -1.0      # remaining time
        f[:, 1] = u         # state dynamics

        ######################
        # Diffusion
        ######################
        Ns = X.shape[0]
        sigma = np.zeros((Ns, self.state_dim), dtype=np.float32)
        sigma[:, 1] = self.sigma_val

        return (f[0], sigma[0]) if single else (f, sigma)

    ################################################
    # for RL training
    ################################################
    def reset(self):
        """
        Sample initial state from non-target region.
        """
        T = self.Tmax
        x = np.random.uniform(self.x_min, self.x_target - 0.2)
        self.state = np.array([T, x], dtype=np.float32)
        return self.state

    def step(self, action):
        T = self.state[0]
        x = self.state[1]

        # terminal condition at current state
        isTimeOver = (T < self.dt)
        isTarget   = (x >= self.x_target)
        done       = isTimeOver or isTarget

        # if already done, return as-is
        if done:
            reward = 1.0 if isTarget else 0.0
            info = {"is_target": isTarget, "is_time_over": isTimeOver}
            return self.state.copy(), reward, done, info

        # Euler-Maruyama
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -self.u_max, self.u_max)

        f, sigma = self.drift_and_diffusion(self.state, action)

        noise = np.random.randn(self.state_dim).astype(np.float32)
        next_state = self.state + self.dt * f + np.sqrt(self.dt) * sigma * noise
        self.state = next_state.astype(np.float32)

        # terminal condition after transition
        T_next = self.state[0]
        x_next = self.state[1]

        isTimeOver = (T_next < self.dt)
        isTarget   = (x_next >= self.x_target)
        done       = isTimeOver or isTarget

        reward = 1.0 if isTarget else 0.0
        info = {"is_target": isTarget, "is_time_over": isTimeOver}

        return self.state.copy(), reward, done, info

    ################################################
    # for PINN training
    ################################################
    def sample_pinn_collocation_points(self, nPDE, nTarget, nAvoid):
        """
        Returns:
            X_pde  : interior points in (0,Tmax] x (x_min, x_target)
            X_tgt  : Target boundary at x=x_target
            X_avoid: lateral boundary at T=0 
        """

        def lhs_box(dim, num, lb, ub):
            u = qmc.LatinHypercube(d=dim).random(num)
            return qmc.scale(u, lb, ub).astype(np.float32)

        # Interior points
        X_pde = lhs_box(
            2, nPDE,
            [0.0, self.x_min],
            [self.Tmax, self.x_target]
        )

        # Lateral boundary at x = x_target
        T_tgt = lhs_box(1, nTarget, [0.0], [self.Tmax]).reshape(-1)
        x_tgt = self.x_target * np.ones(nTarget, dtype=np.float32)
        X_tgt = np.column_stack([T_tgt, x_tgt]).astype(np.float32)

        # Initial condition at T=0
        x_avoid = lhs_box(1, nAvoid, [self.x_min], [self.x_target]).reshape(-1)
        T_avoid = np.zeros(nAvoid, dtype=np.float32)
        X_avoid = np.column_stack([T_avoid, x_avoid]).astype(np.float32)


        return X_pde, X_tgt, X_avoid

    def evaluate_physics_model(self, X_pde, U_pde):
        f, sigma = self.drift_and_diffusion(X_pde, U_pde)
        diag = True
        return f, sigma, diag

    @classmethod
    def make_eval_states(cls, T, num_grid):
        x = np.linspace(-2.0, 2.0, num_grid)
        T_arr = np.full_like(x, T)
        states = np.stack([T_arr, x], axis=-1).reshape(-1, 2)
        meta = {"x": x, "T": T}
        return states, meta

    @classmethod
    def plot_values(cls, value, meta):
        
        x = meta["x"]
        T_eval = meta["T"]
        value_true = true_reach_probability(T_eval, x, mu=cls.u_max, sigma=cls.sigma_val, x_target=cls.x_target)

       
        plt.figure(figsize=(8, 5))
        plt.plot(x, value)       
        plt.plot(x, value_true, "--", label="Ground truth")        
        plt.xlabel("x")
        plt.ylabel("Reach probability")
        plt.legend()
        plt.grid(True)
        plt.show()
    

def true_reach_probability(T, x, mu=1.0, sigma=1.0, x_target=2.0):
    """
    Finite-horizon reach probability:
        P_x( sup_{0<=s<=T} X_s >= x_target )
    for dX = mu dt + sigma dW
    """
    x = np.asarray(x, dtype=np.float64)

    # T=0 の場合を安全に処理
    if T <= 0:
        return (x >= x_target).astype(np.float64)

    prob = np.ones_like(x, dtype=np.float64)
    mask = x < x_target

    z1 = (x[mask] - x_target + mu * T) / (sigma * np.sqrt(T))
    z2 = (x[mask] - x_target - mu * T) / (sigma * np.sqrt(T))

    prob[mask] = norm.cdf(z1) + np.exp(2.0 * mu * (x_target - x[mask]) / (sigma**2)) * norm.cdf(z2)
    prob = np.clip(prob, 0.0, 1.0)

    return prob




    
    

##########################################################
# Test env implementation
##########################################################
if __name__ == "__main__":

    rlenv = Env()

    ############################
    # Common dynamics
    ############################
    x = [1.0, 0.0]   # [T, x]
    u = [1.0]
    f, sigma = rlenv.drift_and_diffusion(x, u)
    print("f:", f)
    print("sigma:", sigma)

    X = [[2.0, -1.0], [1.0, 0.0], [0.5, 1.5]]
    U = [[1.0], [0.0], [-0.5]]
    F, SIGMA = rlenv.drift_and_diffusion(X, U)
    print("F:", F)
    print("SIGMA:", SIGMA)
    print("============")

    ############################
    # RL steps
    ############################
    state = rlenv.reset()
    print("initial:", np.round(state, 3))
    for t in range(50):
        next_state, reward, done, info = rlenv.step([1.0])
        print(f"t={t+1} state:", np.round(next_state, 3), "reward:", reward, "done:", done, info)
        if done:
            print("terminated")
            break
    print("============")

    ############################
    # PINN functions
    ############################
    X_pde, X_tgt, X_avoid = rlenv.sample_pinn_collocation_points(
        nPDE=5, nTarget=4, nAvoid=4
    )
    print("Xpde: [T, x]\n", np.round(X_pde, 3))
    print("Xtarget: [T, x]\n", np.round(X_tgt, 3))
    print("Xavoid: [T, x]\n", np.round(X_avoid, 3))

    U_pde = np.ones((X_pde.shape[0], 1), dtype=np.float32)
    f, sigma, diag = rlenv.evaluate_physics_model(X_pde, U_pde)
    print("f:", f)
    print("sigma:", sigma)
    print("diag:", diag)
    
    
    