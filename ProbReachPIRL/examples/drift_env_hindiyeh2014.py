# -*- coding: utf-8 -*-
"""
Reach-avoid environment for the drift-equilibrium problem in
Hindiyeh and Gerdes (2014).

State:
    X = [T, beta, r, Ux]

where
    T    : remaining time [s]
    beta : vehicle sideslip angle at CG [rad]
    r    : yaw rate [rad/s]
    Ux   : longitudinal velocity [m/s]

Action:
    U = [delta, FxR]

where
    delta : front steering angle [rad]
    FxR   : rear longitudinal drive force [N]

This implementation intentionally follows the three-state RWD
bicycle model used in the manuscript, rather than the path-relative
7-state bicycle model.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc


class Env(object):
    def __init__(self):
        # ------------------------------------------------------------
        # Basic dimensions
        # ------------------------------------------------------------
        self.dt = 0.02
        self.Tmax = 2.0

        # X = [T, beta, r, Ux]
        self.state_dim = 4
        self.vehicle_state_dim = 3

        # U = [delta, FxR]
        self.action_dim = 2

        # ------------------------------------------------------------
        # Vehicle parameters from Hindiyeh and Gerdes (2014)
        # ------------------------------------------------------------
        self.m = 1724.0
        self.Iz = 1300.0
        self.a = 1.35       # CG to front axle [m]
        self.b = 1.15       # CG to rear axle [m]
        self.g = 9.81

        self.CaF = 120000.0 # front cornering stiffness [N/rad]
        self.CaR = 175000.0 # rear cornering stiffness [N/rad]
        self.mu_ref = 0.55  # gravel-surface friction coefficient

        # Static normal loads
        self.FzF = self.m * self.g * self.b / (self.a + self.b)
        self.FzR = self.m * self.g * self.a / (self.a + self.b)

        # ------------------------------------------------------------
        # Input constraints
        # ------------------------------------------------------------
        self.delta_max = np.deg2rad(23.0)
        self.FxR_min = 0.0
        self.FxR_max = 0.999 * self.mu_ref * self.FzR

        # ------------------------------------------------------------
        # State constraints and sampling ranges
        # ------------------------------------------------------------
        self.beta_max = np.deg2rad(35.0)
        self.r_max = 1.6
        self.Ux_min = 4.0
        self.Ux_max = 12.0

        self._setup_scaling()

        # ------------------------------------------------------------
        # Diffusion coefficients for stochastic reach-avoid
        # ------------------------------------------------------------
        self.sigma_T = 0.0
        self.sigma_beta = np.deg2rad(0.5)
        self.sigma_r = 0.03
        self.sigma_Ux = 0.10

        # ------------------------------------------------------------
        # Drift equilibrium target
        # The paper fixes Ux_eq and delta_eq and solves for beta_eq,
        # r_eq, and FxR_eq.
        # ------------------------------------------------------------
        self.set_drift_equilibrium(
            Ux_eq=8.0,
            delta_eq=np.deg2rad(-12.0),
            mu=self.mu_ref,
            verbose=0,
        )

        # Reach-set tolerances around the drift equilibrium
        self.beta_tol = np.deg2rad(4.0)
        self.r_tol = 0.12
        self.Ux_tol = 0.5

    ################################################
    # Scaling utilities
    ################################################
    def _setup_scaling(self):
        self.X_low = np.array([
            0.0,
            -self.beta_max,
            -self.r_max,
            self.Ux_min,
        ], dtype=np.float32)

        self.X_high = np.array([
            self.Tmax,
            self.beta_max,
            self.r_max,
            self.Ux_max,
        ], dtype=np.float32)

        self.U_low = np.array([
            -self.delta_max,
            self.FxR_min,
        ], dtype=np.float32)

        self.U_high = np.array([
            self.delta_max,
            self.FxR_max,
        ], dtype=np.float32)

        self.X_center = 0.5 * (self.X_high + self.X_low)
        self.X_scale = 0.5 * (self.X_high - self.X_low)

        self.U_center = 0.5 * (self.U_high + self.U_low)
        self.U_scale = 0.5 * (self.U_high - self.U_low)

    def scale_state(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (X - self.X_center) / self.X_scale

    def unscale_state(self, Xs):
        Xs = np.asarray(Xs, dtype=np.float32)
        return self.X_center + self.X_scale * Xs

    def scale_action(self, U):
        U = np.asarray(U, dtype=np.float32)
        return (U - self.U_center) / self.U_scale

    def unscale_action(self, Us):
        Us = np.asarray(Us, dtype=np.float32)
        return self.U_center + self.U_scale * Us

    def _clip_action(self, U):
        U = np.asarray(U, dtype=np.float32)
        single = (U.ndim == 1)
        if single:
            U = U[None, :]

        Uc = U.copy()
        Uc[:, 0] = np.clip(Uc[:, 0], -self.delta_max, self.delta_max)
        Uc[:, 1] = np.clip(Uc[:, 1], self.FxR_min, self.FxR_max)

        return Uc[0] if single else Uc

    ################################################
    # Tire model: modified Fiala with rear-force derating
    ################################################
    def fiala_lateral_force(self, alpha, Fz, Ca, mu, n=1.0):
        """
        Modified Fiala lateral tire force model.

        The sign convention follows the manuscript expression:
            z = tan(alpha)
            Fy = Ca*z - Ca^2/(3 n mu Fz) |z| z
                 + Ca^3/(27 n^2 mu^2 Fz^2) z^3
        before saturation at n mu Fz sign(alpha).

        The model is odd in alpha and saturates smoothly at the
        available lateral force.
        """
        alpha = np.asarray(alpha, dtype=np.float32)
        n = np.asarray(n, dtype=np.float32)
        n = np.maximum(n, 1.0e-4)

        z = np.tan(alpha)
        alpha_sl = np.arctan(3.0 * n * mu * Fz / Ca)
        Fy_unsat = (
            Ca * z
            - (Ca ** 2) / (3.0 * n * mu * Fz) * np.abs(z) * z
            + (Ca ** 3) / (27.0 * (n ** 2) * (mu ** 2) * (Fz ** 2)) * (z ** 3)
        )
        Fy_sat = n * mu * Fz * np.sign(alpha)
        return np.where(np.abs(alpha) < alpha_sl, Fy_unsat, Fy_sat).astype(np.float32)

    def tire_forces(self, beta, r, Ux, delta, FxR, mu=None):
        """
        Compute front/rear lateral tire forces for the three-state model.
        """
        if mu is None:
            mu = self.mu_ref

        Ux = np.maximum(Ux, 1.0e-3)
        Uy = Ux * beta

        alpha_f = delta - np.arctan2(Uy + self.a * r, Ux)
        alpha_r = -np.arctan2(Uy - self.b * r, Ux)

        FxR = np.clip(FxR, self.FxR_min, 0.999 * mu * self.FzR)
        nR = np.sqrt(np.maximum((mu * self.FzR) ** 2 - FxR ** 2, 0.0)) / (mu * self.FzR)

        FyF = self.fiala_lateral_force(alpha_f, self.FzF, self.CaF, mu, n=1.0)
        FyR = self.fiala_lateral_force(alpha_r, self.FzR, self.CaR, mu, n=nR)

        return FyF, FyR

    ################################################
    # Dynamics model
    ################################################
    def drift_and_diffusion(self, X, U):
        """
        Continuous-time stochastic dynamics:
            dX = f(X, U) dt + sigma dW

        X: [T, beta, r, Ux]
        U: [delta, FxR]
        """
        X = np.asarray(X, dtype=np.float32)
        U = np.asarray(U, dtype=np.float32)

        single = (X.ndim == 1)
        if single:
            X = X[None, :]
            U = U[None, :]

        U = self._clip_action(U)

        beta = X[:, 1]
        r = X[:, 2]
        Ux = np.maximum(X[:, 3], 1.0e-3)

        delta = U[:, 0]
        FxR = U[:, 1]

        FyF, FyR = self.tire_forces(beta, r, Ux, delta, FxR, mu=self.mu_ref)

        f = np.zeros_like(X, dtype=np.float32)
        f[:, 0] = -1.0
        f[:, 1] = (FyF + FyR) / (self.m * Ux) - r
        f[:, 2] = (self.a * FyF - self.b * FyR) / self.Iz
        f[:, 3] = (FxR - FyF * np.sin(delta)) / self.m + r * Ux * beta

        sigma = np.zeros_like(X, dtype=np.float32)
        sigma[:, 0] = self.sigma_T
        sigma[:, 1] = self.sigma_beta
        sigma[:, 2] = self.sigma_r
        sigma[:, 3] = self.sigma_Ux

        return (f[0], sigma[0]) if single else (f, sigma)

    ################################################
    # Equilibrium calculation
    ################################################
    def find_drift_equilibrium(self, Ux_eq=8.0, delta_eq=np.deg2rad(-12.0), mu=None, verbose=0):
        """
        Solve Eq. (4)-type equilibrium conditions for fixed Ux_eq
        and delta_eq.

        Unknown:
            z = [beta_eq, r_eq, FxR_eq]
        """
        if mu is None:
            mu = self.mu_ref

        z_low = np.array([
            -self.beta_max,
            -self.r_max,
            self.FxR_min,
        ], dtype=np.float64)

        z_high = np.array([
            self.beta_max,
            self.r_max,
            0.999 * mu * self.FzR,
        ], dtype=np.float64)

        z_center = 0.5 * (z_high + z_low)
        z_scale = 0.5 * (z_high - z_low)

        def unscale_z(zs):
            return z_center + z_scale * zs

        def residual_scaled(zs):
            beta_eq, r_eq, FxR_eq = unscale_z(zs)

            X = np.array([1.0, beta_eq, r_eq, Ux_eq], dtype=np.float32)
            U = np.array([delta_eq, FxR_eq], dtype=np.float32)
            f, _ = self.drift_and_diffusion(X, U)

            return np.array([f[1], f[2], f[3]], dtype=np.float64)

        # Initial guess close to the drift equilibrium reported in the paper.
        z0 = np.array([
            np.deg2rad(-20.44),
            0.600,
            2293.0,
        ], dtype=np.float64)
        z0_scaled = (z0 - z_center) / z_scale
        z0_scaled = np.clip(z0_scaled, -0.95, 0.95)

        sol_scaled = least_squares(
            residual_scaled,
            z0_scaled,
            bounds=(-np.ones_like(z0_scaled), np.ones_like(z0_scaled)),
            jac="3-point",
            xtol=1.0e-10,
            ftol=1.0e-10,
            gtol=1.0e-10,
            max_nfev=2000,
            verbose=verbose,
        )

        res_inf = np.linalg.norm(sol_scaled.fun, ord=np.inf)
        if res_inf > 1.0e-5:
            raise RuntimeError(
                "No accurate drift equilibrium found. "
                f"residual_inf_norm={res_inf:.3e}, "
                f"Ux_eq={Ux_eq}, delta_eq={delta_eq}, mu={mu}"
            )

        sol = sol_scaled
        sol["x_scaled"] = sol_scaled.x.copy()
        sol["x"] = unscale_z(sol_scaled.x)

        if verbose >= 1:
            print(f"Zeq [beta, r, FxR]: {sol['x']}")

        return sol

    def set_drift_equilibrium(self, Ux_eq=8.0, delta_eq=np.deg2rad(-12.0), mu=None, verbose=0):
        if mu is None:
            mu = self.mu_ref

        self.Ux_eq = float(Ux_eq)
        self.delta_eq = float(delta_eq)
        self.mu_eq = float(mu)

        sol = self.find_drift_equilibrium(
            Ux_eq=self.Ux_eq,
            delta_eq=self.delta_eq,
            mu=self.mu_eq,
            verbose=verbose,
        )

        self.beta_eq = float(sol.x[0])
        self.r_eq = float(sol.x[1])
        self.FxR_eq = float(sol.x[2])

        Xd = np.array([1.0, self.beta_eq, self.r_eq, self.Ux_eq], dtype=np.float32)
        Ud = np.array([self.delta_eq, self.FxR_eq], dtype=np.float32)
        fd, _ = self.drift_and_diffusion(Xd, Ud)

        self.drift_equilibrium_solution = (Xd, Ud)
        self.drift_equilibrium_residual = fd.copy()

        return sol

    ################################################
    # Reach-avoid sets
    ################################################
    def is_target(self, X):
        X = np.asarray(X, dtype=np.float32)
        single = (X.ndim == 1)
        if single:
            X = X[None, :]

        beta = X[:, 1]
        r = X[:, 2]
        Ux = X[:, 3]

        mask = (
            (np.abs(beta - self.beta_eq) <= self.beta_tol)
            & (np.abs(r - self.r_eq) <= self.r_tol)
            & (np.abs(Ux - self.Ux_eq) <= self.Ux_tol)
        )

        return bool(mask[0]) if single else mask

    def is_unsafe(self, X):
        X = np.asarray(X, dtype=np.float32)
        single = (X.ndim == 1)
        if single:
            X = X[None, :]

        beta = X[:, 1]
        r = X[:, 2]
        Ux = X[:, 3]

        mask = (
            (np.abs(beta) > self.beta_max)
            | (np.abs(r) > self.r_max)
            | (Ux < self.Ux_min)
            | (Ux > self.Ux_max)
        )

        return bool(mask[0]) if single else mask

    ################################################
    # Methods for RL training
    ################################################
    def reset(self):
        T = self.Tmax
        beta = self.beta_eq + np.random.uniform(np.deg2rad(-10.0), np.deg2rad(10.0))
        r = self.r_eq + np.random.uniform(-0.35, 0.35)
        Ux = self.Ux_eq + np.random.uniform(-1.5, 1.5)

        self.state = np.array([T, beta, r, Ux], dtype=np.float32)
        return self.scale_state(self.state)

    def step(self, action):
        """
        One Euler-Maruyama step.

        The agent provides a scaled action in [-1, 1]^action_dim.
        The returned state is also scaled.
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action_phys = self.unscale_action(action)
        action_phys = self._clip_action(action_phys)

        T = self.state[0]
        is_time_over = (T < self.dt)
        is_target = self.is_target(self.state)
        is_unsafe = self.is_unsafe(self.state)
        done = is_time_over or is_target or is_unsafe

        if done:
            reward = 1.0 if is_target else 0.0
            info = {
                "is_time_over": is_time_over,
                "is_target": is_target,
                "is_unsafe": is_unsafe,
            }
            return self.scale_state(self.state.copy()), reward, done, info

        f, sigma = self.drift_and_diffusion(self.state, action_phys)
        noise = np.random.randn(self.state_dim).astype(np.float32)
        next_state = self.state + self.dt * f + np.sqrt(self.dt) * sigma * noise
        next_state[0] = max(next_state[0], 0.0)

        self.state = next_state.astype(np.float32)

        is_time_over = (self.state[0] < self.dt)
        is_target = self.is_target(self.state)
        is_unsafe = self.is_unsafe(self.state)
        done = is_time_over or is_target or is_unsafe

        reward = 1.0 if is_target else 0.0
        info = {
            "is_time_over": is_time_over,
            "is_target": is_target,
            "is_unsafe": is_unsafe,
        }

        return self.scale_state(self.state.copy()), reward, done, info

    ################################################
    # PINN utilities
    ################################################
    def sample_pinn_collocation_points(self, nPDE, nTarget, nAvoid):
        def lhs_box(dim, num, lb, ub):
            u = qmc.LatinHypercube(d=dim).random(num)
            return qmc.scale(u, lb, ub).astype(np.float32)

        lb_pde = [0.0, -self.beta_max, -self.r_max, self.Ux_min]
        ub_pde = [self.Tmax, self.beta_max, self.r_max, self.Ux_max]
        X_pde = lhs_box(self.state_dim, nPDE, lb_pde, ub_pde)

        # Target boundary around the drift equilibrium for all T.
        T = np.random.uniform(0.0, self.Tmax, size=nTarget)
        beta = np.random.uniform(self.beta_eq - self.beta_tol, self.beta_eq + self.beta_tol, size=nTarget)
        r = np.random.uniform(self.r_eq - self.r_tol, self.r_eq + self.r_tol, size=nTarget)
        Ux = np.random.uniform(self.Ux_eq - self.Ux_tol, self.Ux_eq + self.Ux_tol, size=nTarget)
        X_tgt = np.column_stack([T, beta, r, Ux]).astype(np.float32)

        # Avoid boundary points: sideslip, yaw-rate, or speed limits.
        X_avoid = lhs_box(self.state_dim, nAvoid, lb_pde, ub_pde)
        modes = np.arange(nAvoid) % 4

        idx = modes == 0
        X_avoid[idx, 1] = self.beta_max * np.where(np.random.rand(np.sum(idx)) > 0.5, 1.0, -1.0)

        idx = modes == 1
        X_avoid[idx, 2] = self.r_max * np.where(np.random.rand(np.sum(idx)) > 0.5, 1.0, -1.0)

        idx = modes == 2
        X_avoid[idx, 3] = self.Ux_min

        idx = modes == 3
        X_avoid[idx, 3] = self.Ux_max

        return (
            self.scale_state(X_pde),
            self.scale_state(X_tgt),
            self.scale_state(X_avoid.astype(np.float32)),
        )

    def evaluate_physics_model(self, X_pde, U_pde):
        """
        X_pde and U_pde are scaled variables.

        Returns dynamics in scaled coordinates:
            dXs = fs dt + sigmas dW
        """
        X_phys = self.unscale_state(X_pde)
        U_phys = self.unscale_action(U_pde)

        f_phys, sigma_phys = self.drift_and_diffusion(X_phys, U_phys)

        f_scaled = f_phys / self.X_scale
        sigma_scaled = sigma_phys / self.X_scale

        diag = True
        return f_scaled, sigma_scaled, diag

    ################################################
    # Evaluation states for visualization
    ################################################
    def make_eval_states(self, T, num_grid, plane="beta_r"):
        if plane == "beta_r":
            beta_grid = np.linspace(-self.beta_max, self.beta_max, num=num_grid)
            r_grid = np.linspace(-self.r_max, self.r_max, num=num_grid)
            BETA, R = np.meshgrid(beta_grid, r_grid, indexing="ij")

            Ts = np.full_like(BETA, T)
            UX = np.full_like(BETA, self.Ux_eq)

            state = np.stack([Ts, BETA, R, UX], axis=-1).reshape(-1, self.state_dim)
            meta = {
                "x": beta_grid,
                "y": r_grid,
                "xlabel": "beta [rad]",
                "ylabel": "r [rad/s]",
                "T": T,
                "plane": plane,
            }
            return state.astype(np.float32), meta

        elif plane == "beta_Ux":
            beta_grid = np.linspace(-self.beta_max, self.beta_max, num=num_grid)
            Ux_grid = np.linspace(self.Ux_min, self.Ux_max, num=num_grid)
            BETA, UX = np.meshgrid(beta_grid, Ux_grid, indexing="ij")

            Ts = np.full_like(BETA, T)
            R = np.full_like(BETA, self.r_eq)

            state = np.stack([Ts, BETA, R, UX], axis=-1).reshape(-1, self.state_dim)
            meta = {
                "x": beta_grid,
                "y": Ux_grid,
                "xlabel": "beta [rad]",
                "ylabel": "Ux [m/s]",
                "T": T,
                "plane": plane,
            }
            return state.astype(np.float32), meta

        else:
            raise ValueError("plane must be either 'beta_r' or 'beta_Ux'.")


if __name__ == "__main__":
    rlenv = Env()

    print("Drift equilibrium")
    print("beta_eq [deg]:", np.rad2deg(rlenv.beta_eq))
    print("r_eq [rad/s]:", rlenv.r_eq)
    print("Ux_eq [m/s]:", rlenv.Ux_eq)
    print("delta_eq [deg]:", np.rad2deg(rlenv.delta_eq))
    print("FxR_eq [N]:", rlenv.FxR_eq)
    print("residual:", rlenv.drift_equilibrium_residual)

    X_pde, X_tgt, X_avoid = rlenv.sample_pinn_collocation_points(5, 3, 4)
    U_pde = np.zeros((X_pde.shape[0], rlenv.action_dim), dtype=np.float32)
    f, sigma, diag = rlenv.evaluate_physics_model(X_pde, U_pde)
    print("X_pde shape:", X_pde.shape)
    print("f shape:", f.shape)
    print("sigma shape:", sigma.shape)
    print("diag:", diag)
