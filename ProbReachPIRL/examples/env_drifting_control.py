# -*- coding: utf-8 -*-
"""
Dynamic bicycle reach-avoid environment for PIRL/PINN experiments.

State vector follows the same convention as the 2D example:
    X = [T, ey, epsi, vx, vy, r, delta, mu]

where the physical/augmented vehicle state is 7-dimensional:
    [ey, epsi, vx, vy, r, delta, mu]

Action:
    U = [delta_dot_cmd, Fx_cmd]

Reach set:
    neighborhood of a drift equilibirium

Avoid set:
    road departure, excessive sideslip, excessive yaw rate,
    or loss of forward speed.
"""
import os
import numpy as np
from scipy.stats import qmc
from scipy.optimize import least_squares

class Env(object):
    def __init__(self):
        # ------------------------------------------------------------
        # Basic dimensions
        # ------------------------------------------------------------
        self.dt = float(os.environ.get("DRIFT_DT", "0.02"))
        self.Tmax = 5.0

        # X = [T, ey, epsi, vx, vy, r, delta, mu]
        self.state_dim = 8
        self.vehicle_state_dim = 7

        # U = [delta_dot, Fx]
        self.action_dim = 2

        # ------------------------------------------------------------
        # Vehicle parameters: Hindiyeh and Gerdes (2014), P1 testbed
        # ------------------------------------------------------------
        self.m = 1724.0       # mass [kg]
        self.Iz = 1300.0      # yaw inertia [kg m^2]
        self.lf = 1.35        # CG to front axle a [m]
        self.lr = 1.15        # CG to rear axle b [m]
        self.g = 9.81

        # Lumped axle cornering stiffness parameters [N/rad]
        self.Cf = 120000.0
        self.Cr = 175000.0

        # ------------------------------------------------------------
        # Input constraints
        # ------------------------------------------------------------
        self.delta_dot_max = 1.5    # [rad/s]
        self.Fx_max = 5000.0        # [N]
        self.Fx_min = 1000.0        # [N]

        # ------------------------------------------------------------
        # State constraints and sampling ranges
        # ------------------------------------------------------------
        self.ey_max = 2.0
        self.epsi_max = 0.8
        self.vx_min = 3.0
        self.vx_max = 25.0
        self.vy_max = 12.0
        self.r_max = 2.5
        self.delta_max = 0.7
        self.mu_min = 0.25
        self.mu_max = 1.05

        # Spin / loss-of-control threshold
        self.beta_max = np.deg2rad(45.0)

        # Setup scaling
        self._setup_scaling()

        # ------------------------------------------------------------
        # Diffusion coefficients for stochastic reach-avoid
        # Diagonal noise model. T and mu are deterministic by default.
        # ------------------------------------------------------------
        self.sigma_T = 0.0
        self.sigma_ey = 0.03
        self.sigma_epsi = 0.015
        self.sigma_vx = 0.20
        self.sigma_vy = 0.20
        self.sigma_r = 0.05
        self.sigma_delta = 0.01
        self.sigma_mu = 0.0

        # Training distribution controls.  scale=1 samples nearly the full
        # plotting box for ey, epsi, beta, and r; scale=0 collapses to the
        # mu-dependent drift equilibrium.
        self.reset_scale = float(os.environ.get("DRIFT_RESET_SCALE", "1.0"))
        self.reset_mode = os.environ.get("DRIFT_RESET_MODE", "full").lower()
        self.reset_mixture_probs = self._parse_reset_mixture_probs(
            os.environ.get("DRIFT_RESET_MIXTURE_PROBS", "0.45,0.45,0.10")
        )
        self.reset_t_mode = os.environ.get("DRIFT_RESET_T_MODE", "fixed").lower()
<<<<<<< HEAD
        self.reset_t_min = float(os.environ.get("DRIFT_RESET_T_MIN", "0.0"))
=======
        self.reset_t_min = float(os.environ.get("DRIFT_RESET_T_MIN", "0.2"))
        self.reset_t_max = float(os.environ.get("DRIFT_RESET_T_MAX", str(self.Tmax)))
>>>>>>> 84a830a96512b2b8792fa027d09917d7db538f46
        self.reset_epsi_min = -0.2
        self.reset_epsi_max = 0.8

        # ------------------------------------------------------------
        # Drift equilibrium target
        # ------------------------------------------------------------
        # We specify a fixed path curvature and solve drift equilibria
        # over the episode-level friction range.  The remaining target
        # quantities are interpolated as functions of the current mu.
        # Curvature chosen so that the path-relative equilibrium is
        # consistent with the paper's drift point:
        # beta ~= -20.44 deg, r ~= 0.600 rad/s, Ux = 8 m/s.
        self.kappa_ref = 0.600 * np.cos(np.deg2rad(-20.44)) / 8.0
        self.vx_d = 8.0
        self.mu_d = 0.55
        self.mu_target_min = 0.50 #0.40
        self.mu_target_max = 0.60 #0.80
        self.mu_grid = np.linspace(self.mu_target_min, self.mu_target_max, 17, dtype=np.float32)
        self.build_drift_equilibrium_table(
            vx_d=self.vx_d,
            mu_grid=self.mu_grid,
            nominal_mu=self.mu_d,
            verbose=0,
        )

        # Target tolerances
        self.ey_tol = 0.2
        self.epsi_tol = np.deg2rad(4.0)
        self.vx_tol = 0.5
        self.beta_tol = np.deg2rad(4.0)
        self.r_tol = 0.2
        self.delta_tol = np.deg2rad(10.0)

    ################################################
    # Utility functions
    ################################################
    def _setup_scaling(self):
        self.X_low = np.array([
            0.0,
            -self.ey_max,
            -self.epsi_max,
            self.vx_min,
            -self.vy_max,
            -self.r_max,
            -self.delta_max,
            self.mu_min,
        ], dtype=np.float32)
    
        self.X_high = np.array([
            self.Tmax,
            self.ey_max,
            self.epsi_max,
            self.vx_max,
            self.vy_max,
            self.r_max,
            self.delta_max,
            self.mu_max,
        ], dtype=np.float32)
    
        self.U_low = np.array([
            -self.delta_dot_max,
            self.Fx_min,
        ], dtype=np.float32)
    
        self.U_high = np.array([
            self.delta_dot_max,
            self.Fx_max,
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
        Uc[:, 0] = np.clip(Uc[:, 0], -self.delta_dot_max, self.delta_dot_max)
        Uc[:, 1] = np.clip(Uc[:, 1], self.Fx_min, self.Fx_max)

        return Uc[0] if single else Uc

    def beta(self, X):
        """Sideslip angle beta = atan2(vy, vx)."""
        X = np.asarray(X, dtype=np.float32)
        single = (X.ndim == 1)
        if single:
            X = X[None, :]

        vx = np.maximum(X[:, 3], 1.0e-3)
        vy = X[:, 4]
        b = np.arctan2(vy, vx)
        return b[0] if single else b

    def is_target(self, X):
        """Return boolean mask indicating the reach set."""
        X = np.asarray(X, dtype=np.float32)
        single = (X.ndim == 1)
        if single:
            X = X[None, :]

        ey = X[:, 1]
        epsi = X[:, 2]
        vx = X[:, 3]
        r = X[:, 5]
        delta = X[:, 6]
        mu = X[:, 7]
        beta = self.beta(X)
        target = self.get_drift_target(mu)

        mask = (
            (mu >= self.mu_target_min)
            & (mu <= self.mu_target_max)
            & (np.abs(ey - target["ey"]) <= self.ey_tol)
            & (np.abs(epsi - target["epsi"]) <= self.epsi_tol)
            & (np.abs(vx - target["vx"]) <= self.vx_tol)
            & (np.abs(beta - target["beta"]) <= self.beta_tol)
            & (np.abs(r - target["r"]) <= self.r_tol)
            & (np.abs(delta - target["delta"]) <= self.delta_tol)
        )

        return bool(mask[0]) if single else mask

    def is_unsafe(self, X):
        """Return boolean mask indicating the avoid set."""
        X = np.asarray(X, dtype=np.float32)
        single = (X.ndim == 1)
        if single:
            X = X[None, :]

        ey = X[:, 1]
        epsi = X[:, 2]
        vx = X[:, 3]
        vy = X[:, 4]
        r = X[:, 5]
        delta = X[:, 6]
        mu = X[:, 7]
        beta = self.beta(X)

        mask = (
            (np.abs(ey) > self.ey_max)
            | (np.abs(epsi) > self.epsi_max)
            | (vx < self.vx_min)
            | (vx > self.vx_max)
            | (np.abs(vy) > self.vy_max)
            | (np.abs(r) > self.r_max)
            | (np.abs(delta) > self.delta_max)
            | (np.abs(beta) > self.beta_max)
            | (mu < self.mu_min)
            | (mu > self.mu_max)
        )

        return bool(mask[0]) if single else mask

    ################################################
    # Dynamics model (in SI unit)
    ################################################
    def _fiala_lateral_force(self, alpha, C_alpha, Fz, mu, n):
        """Modified Fiala lateral tire force for a lumped axle.

        Parameters
        ----------
        alpha : array_like
            Tire slip angle [rad]. Positive alpha gives positive lateral force,
            consistent with the sign convention used in this environment.
        C_alpha : float
            Cornering stiffness [N/rad].
        Fz : float
            Static normal load [N].
        mu : array_like
            Road friction coefficient.
        n : array_like
            Derating factor due to longitudinal force. n=1 for the front tire.
        """
        alpha = np.asarray(alpha, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        n = np.asarray(n, dtype=np.float64)

        D = np.maximum(n * mu * Fz, 1.0e-9)
        z = np.tan(alpha)
        z_sl = 3.0 * D / C_alpha

        Fy_unsat = (
            C_alpha * z
            - (C_alpha ** 2) / (3.0 * D) * np.abs(z) * z
            + (C_alpha ** 3) / (27.0 * D ** 2) * z ** 3
        )
        Fy_sat = D * np.sign(alpha)
        Fy = np.where(np.abs(z) < z_sl, Fy_unsat, Fy_sat)
        return Fy.astype(np.float32)

    def drift_and_diffusion(self, X, U):
        """
        Continuous-time stochastic dynamics:
            dX = f(X, U) dt + sigma(X, U) dW

        X: shape (state_dim,) or (N, state_dim)
           [T, ey, epsi, vx, vy, r, delta, mu]
        U: shape (action_dim,) or (N, action_dim)
           [delta_dot, Fx]
        """
        X = np.asarray(X, dtype=np.float32)
        U = np.asarray(U, dtype=np.float32)

        single = (X.ndim == 1)
        if single:
            X = X[None, :]
            U = U[None, :]

        U = self._clip_action(U)

        ey = X[:, 1]
        epsi = X[:, 2]
        vx = np.maximum(X[:, 3], 1.0e-3)
        vy = X[:, 4]
        r = X[:, 5]
        delta = X[:, 6]
        mu = np.clip(X[:, 7], self.mu_min, self.mu_max)

        delta_dot = U[:, 0]
        Fx = U[:, 1]

        # ------------------------------------------------------------
        # Tire slip angles
        # ------------------------------------------------------------
        alpha_f = delta - np.arctan2(vy + self.lf * r, vx)
        alpha_r = -np.arctan2(vy - self.lr * r, vx)

        # Static normal loads
        Fzf = self.m * self.g * self.lr / (self.lf + self.lr)
        Fzr = self.m * self.g * self.lf / (self.lf + self.lr)

        # ------------------------------------------------------------
        # Modified Fiala lateral tire model with friction-circle
        # derating for the rear tire.  This follows the modeling
        # structure in Hindiyeh and Gerdes (2014): nF = 1 and
        # nR = sqrt((mu FzR)^2 - FxR^2) / (mu FzR).
        # ------------------------------------------------------------
        eps = 1.0e-9
        Fx_bound = 0.999 * mu * Fzr
        Fx = np.clip(Fx, -Fx_bound, Fx_bound)
        nF = np.ones_like(mu)
        nR = np.sqrt(np.maximum((mu * Fzr) ** 2 - Fx ** 2, 0.0)) / (mu * Fzr + eps)

        Fyf = self._fiala_lateral_force(alpha_f, self.Cf, Fzf, mu, nF)
        Fyr = self._fiala_lateral_force(alpha_r, self.Cr, Fzr, mu, nR)

        # ------------------------------------------------------------
        # Drift term
        # ------------------------------------------------------------
        f = np.zeros_like(X, dtype=np.float32)

        # Remaining-time dynamics
        f[:, 0] = -1.0

        # Path-relative kinematics
        denom = np.maximum(1.0 - self.kappa_ref * ey, 0.2)
        s_dot = (vx * np.cos(epsi) - vy * np.sin(epsi)) / denom
        f[:, 1] = vx * np.sin(epsi) + vy * np.cos(epsi)       # ey_dot
        f[:, 2] = r - self.kappa_ref * s_dot                  # epsi_dot

        # Dynamic bicycle model
        f[:, 3] = (Fx - Fyf * np.sin(delta)) / self.m + vy * r
        f[:, 4] = (Fyr + Fyf * np.cos(delta)) / self.m - vx * r
        f[:, 5] = (self.lf * Fyf * np.cos(delta) - self.lr * Fyr) / self.Iz

        # Steering actuator state and frozen friction parameter
        f[:, 6] = delta_dot
        f[:, 7] = 0.0

        # ------------------------------------------------------------
        # Diffusion: diagonal sigma
        # ------------------------------------------------------------
        Ns = X.shape[0]
        Xdim = X.shape[1]
        sigma = np.zeros((Ns, Xdim), dtype=np.float32)
        sigma[:, 0] = self.sigma_T
        sigma[:, 1] = self.sigma_ey
        sigma[:, 2] = self.sigma_epsi
        sigma[:, 3] = self.sigma_vx
        sigma[:, 4] = self.sigma_vy
        sigma[:, 5] = self.sigma_r
        sigma[:, 6] = self.sigma_delta
        sigma[:, 7] = self.sigma_mu

        return (f[0], sigma[0]) if single else (f, sigma)

    ################################################
    # Equilibrium calculation
    ################################################
    def find_drift_equilibrium(self, vx_d=12.0, mu_d=0.8, verbose=0):
                
        if verbose >= 1:
            print(
                "="*20+"\n",
                "Finding drift equilibrium "
                f"(vx_d={vx_d:.2f} m/s, "
                f"mu_d={mu_d:.2f}, "
                f"R={1.0/self.kappa_ref:.1f} m) ..."
            )
        
        ey_d = 0.0
    
        # z = [beta_d, r_d, delta_d, Fx_d]
        z_low = np.array([
            -self.beta_max,
            -self.r_max,
            -self.delta_max,
            self.Fx_min,
        ], dtype=np.float64)
    
        z_high = np.array([
            self.beta_max,
            self.r_max,
            self.delta_max,
            self.Fx_max,
        ], dtype=np.float64)
    
        z_center = 0.5 * (z_high + z_low)
        z_scale = 0.5 * (z_high - z_low)
    
        def unscale_z(zs):
            return z_center + z_scale * zs
    
        def residual_scaled(zs):
            beta_d, r_d, delta_d, Fx_d = unscale_z(zs) # Unscale
    
            epsi_d = -beta_d
            vy_d = vx_d * np.tan(beta_d)
    
            X = np.array([
                0.85,
                ey_d,
                epsi_d,
                vx_d,
                vy_d,
                r_d,
                delta_d,
                mu_d,
            ], dtype=np.float32)
    
            U = np.array([0.0, Fx_d], dtype=np.float32)
    
            f, _ = self.drift_and_diffusion(X, U)
    
            return np.array([f[2], f[3], f[4], f[5]], dtype=np.float64)

        # Initial guess    
        beta_guess = np.deg2rad(-20.44)
        r_guess = self.kappa_ref * vx_d / max(np.cos(beta_guess), 1.0e-3)
        z0 = np.array([
            beta_guess,
            r_guess,
            np.deg2rad(-12.0),
            2293.0,
        ], dtype=np.float64)    
        z0_scaled = (z0 - z_center) / z_scale
        
        # Sove reast square
        sol_scaled = least_squares(
            residual_scaled,
            z0_scaled,
            bounds=(-np.ones_like(z0_scaled), np.ones_like(z0_scaled)),
            jac="3-point",
            #diff_step=1.0e-4,
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
                f"vx_d={vx_d}, mu_d={mu_d}, kappa_ref={self.kappa_ref}"
            )
        
        # return physical solution
        sol = sol_scaled
        sol["x_scaled"] = sol_scaled.x.copy()
        sol["x"] = unscale_z(sol_scaled.x)

        if verbose >= 1:
            beta_deg = np.rad2deg(sol["x"][0])
            r_eq = sol["x"][1]
            delta_deg = np.rad2deg(sol["x"][2])
            FxR_eq = sol["x"][3]
            print("Drift equilibrium found:")
            print(
                f"  beta = {beta_deg:.2f} deg\n"
                f"  r     = {r_eq:.3f} rad/s\n"
                f"  delta = {delta_deg:.2f} deg\n"
                f"  FxR   = {FxR_eq:.0f} N"
            )
            print(f"  residual_inf = {res_inf:.3e}")

        return sol

    def set_drift_equilibrium(self, vx_d=12.0, mu_d=0.8, verbose=0):
        """Solve and store the drift equilibrium target."""
        self.ey_d = 0.0
        self.vx_d = float(vx_d)
        self.mu_d = float(mu_d)

        sol = self.find_drift_equilibrium(
            vx_d=self.vx_d,
            mu_d=self.mu_d,
            verbose=verbose
        )

        self.beta_d = float(sol.x[0])
        self.epsi_d = -self.beta_d
        self.vy_d = self.vx_d * np.tan(self.beta_d)
        self.r_d = float(sol.x[1])
        self.delta_d = float(sol.x[2])
        self.Fx_d = float(sol.x[3])

        Xd = np.array([
            0.85,
            self.ey_d,
            self.epsi_d,
            self.vx_d,
            self.vy_d,
            self.r_d,
            self.delta_d,
            self.mu_d,
        ], dtype=np.float32)
        Ud = np.array([0.0, self.Fx_d], dtype=np.float32)
        self.drift_equilibrium_solution = (Xd, Ud)
        fd, _ = self.drift_and_diffusion(Xd, Ud)
        self.drift_equilibrium_residual = fd.copy()

        return sol

    def build_drift_equilibrium_table(self, vx_d=8.0, mu_grid=None, nominal_mu=0.55, verbose=0):
        """Solve and store drift equilibrium targets over a fixed-kappa mu grid."""
        if mu_grid is None:
            mu_grid = np.linspace(0.40, 0.80, 9, dtype=np.float32)

        mu_grid = np.asarray(mu_grid, dtype=np.float32)
        if np.any(np.diff(mu_grid) <= 0.0):
            raise ValueError("mu_grid must be strictly increasing.")

        rows = []
        valid_mu = []
        failures = []
        for mu in mu_grid:
            try:
                sol = self.find_drift_equilibrium(vx_d=vx_d, mu_d=float(mu), verbose=verbose)
            except RuntimeError as exc:
                failures.append((float(mu), str(exc)))
                continue

            beta_d = float(sol.x[0])
            valid_mu.append(float(mu))
            rows.append([
                0.0,                         # ey
                -beta_d,                     # epsi
                float(vx_d),                 # vx
                float(vx_d) * np.tan(beta_d),# vy
                beta_d,                      # beta
                float(sol.x[1]),             # r
                float(sol.x[2]),             # delta
                float(sol.x[3]),             # Fx
            ])

        if len(rows) < 2:
            msg = "Need at least two feasible drift equilibria for interpolation."
            if failures:
                msg += " Failed mu values: " + ", ".join(f"{mu:.3f}" for mu, _ in failures)
            raise RuntimeError(msg)

        if verbose >= 1 and failures:
            print(
                "Skipped infeasible drift-equilibrium mu values: "
                + ", ".join(f"{mu:.3f}" for mu, _ in failures)
            )

        mu_grid = np.asarray(valid_mu, dtype=np.float32)
        table = np.asarray(rows, dtype=np.float32)
        self.mu_grid = mu_grid
        self.mu_target_min = float(mu_grid[0])
        self.mu_target_max = float(mu_grid[-1])
        self.drift_equilibrium_table = {
            "mu": mu_grid,
            "ey": table[:, 0],
            "epsi": table[:, 1],
            "vx": table[:, 2],
            "vy": table[:, 3],
            "beta": table[:, 4],
            "r": table[:, 5],
            "delta": table[:, 6],
            "Fx": table[:, 7],
        }

        self.vx_d = float(vx_d)
        self.mu_d = float(nominal_mu)
        nominal = self.get_drift_target(self.mu_d)

        # Nominal attributes are kept for plotting scripts and backward
        # compatibility; target checks use get_drift_target(mu).
        self.ey_d = float(nominal["ey"])
        self.epsi_d = float(nominal["epsi"])
        self.vy_d = float(nominal["vy"])
        self.beta_d = float(nominal["beta"])
        self.r_d = float(nominal["r"])
        self.delta_d = float(nominal["delta"])
        self.Fx_d = float(nominal["Fx"])

        Xd = np.array([
            0.85,
            self.ey_d,
            self.epsi_d,
            self.vx_d,
            self.vy_d,
            self.r_d,
            self.delta_d,
            self.mu_d,
        ], dtype=np.float32)
        Ud = np.array([0.0, self.Fx_d], dtype=np.float32)
        self.drift_equilibrium_solution = (Xd, Ud)
        fd, _ = self.drift_and_diffusion(Xd, Ud)
        self.drift_equilibrium_residual = fd.copy()

    def get_drift_target(self, mu):
        """Return drift target fields interpolated at the given friction coefficient."""
        mu_arr = np.asarray(mu, dtype=np.float32)
        single = (mu_arr.ndim == 0)
        mu_flat = mu_arr.reshape(-1)
        mu_eval = np.clip(mu_flat, self.mu_grid[0], self.mu_grid[-1])

        target = {}
        for key in ["ey", "epsi", "vx", "vy", "beta", "r", "delta", "Fx"]:
            vals = self.drift_equilibrium_table[key]
            interp = np.interp(mu_eval, self.mu_grid, vals).astype(np.float32)
            target[key] = float(interp[0]) if single else interp.reshape(mu_arr.shape)
        return target


    ################################################
    # Methods for RL training
    ################################################
    def _parse_reset_mixture_probs(self, value):
        probs = np.asarray([float(v) for v in value.replace(" ", ",").split(",") if v], dtype=np.float64)
        if probs.shape != (3,):
            raise ValueError(
                "DRIFT_RESET_MIXTURE_PROBS must contain three values: "
                "beta_r,ey_epsi,full."
            )
        if np.any(probs < 0.0) or probs.sum() <= 0.0:
            raise ValueError("DRIFT_RESET_MIXTURE_PROBS must be nonnegative with positive sum.")
        return probs / probs.sum()

    def _sample_reset_component(self, center, low, high, scale):
        """Sample from a range contracted toward center by reset_scale."""
        scale = max(float(scale), 0.0)
        lo = center + min(scale, 1.0) * (low - center)
        hi = center + min(scale, 1.0) * (high - center)
        return np.random.uniform(lo, hi)

    def reset(self):
        """Sample an initial state on a beta-r / ey-epsi box around the drift target."""
        if self.reset_t_mode == "fixed":
            T = self.Tmax
        elif self.reset_t_mode == "random":
<<<<<<< HEAD
            t_min = np.clip(self.reset_t_min, 0.0, self.Tmax)
            T = np.random.uniform(t_min, self.Tmax)
=======
            t_low = max(0.0, min(self.reset_t_min, self.Tmax))
            t_high = max(t_low, min(self.reset_t_max, self.Tmax))
            T = np.random.uniform(t_low, t_high)
>>>>>>> 84a830a96512b2b8792fa027d09917d7db538f46
        else:
            raise ValueError("DRIFT_RESET_T_MODE must be either 'fixed' or 'random'.")

        mu = np.random.uniform(self.mu_target_min, self.mu_target_max)
        target = self.get_drift_target(mu)

        scale = self.reset_scale
        if self.reset_mode == "full":
            reset_plane = "full"
        elif self.reset_mode == "mixture":
            reset_plane = np.random.choice(
                ["beta_r", "ey_epsi", "full"],
                p=self.reset_mixture_probs,
            )
        else:
            raise ValueError("DRIFT_RESET_MODE must be either 'full' or 'mixture'.")

        if reset_plane in ("ey_epsi", "full"):
            ey = self._sample_reset_component(
                target["ey"], -self.ey_max, self.ey_max, scale
            )
            epsi = self._sample_reset_component(
                target["epsi"], self.reset_epsi_min, self.reset_epsi_max, scale
            )
        else:
            ey = target["ey"]
            epsi = target["epsi"]

        if reset_plane in ("beta_r", "full"):
            beta0 = self._sample_reset_component(
                target["beta"], -self.beta_max, self.beta_max, scale
            )
            r = self._sample_reset_component(
                target["r"], -self.r_max, self.r_max, scale
            )
        else:
            beta0 = target["beta"]
            r = target["r"]

        vx = target["vx"]
        vy = vx * np.tan(beta0)
        delta = target["delta"]
        
        self.state = np.array([T, ey, epsi, vx, vy, r, delta, mu], dtype=np.float32)
        return self.scale_state(self.state)

    def step(self, action):
        """One Euler-Maruyama step.
    
        The agent provides a scaled action in [-1, 1]^action_dim.
        The returned state is also scaled.
        Internally, self.state is kept in physical coordinates.
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action_phys = self.unscale_action(action)
        action_phys = self._clip_action(action_phys)
    
        # terminal check uses physical state
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
        next_state[7] = np.clip(next_state[7], self.mu_min, self.mu_max)
    
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
    # for PINN training
    ################################################
    def sample_pinn_collocation_points(self, nPDE, nTarget, nAvoid):
        """
        Return collocation points in the same style as the 2D example:
            X_pde, X_tgt, X_avoid

        X_pde   : interior points for HJB residual
        X_tgt   : target boundary points, value should be 1
        X_avoid : avoid boundary points, value should be 0
        """
        def lhs_box(dim, num, lb, ub):
            u = qmc.LatinHypercube(d=dim).random(num)
            return qmc.scale(u, lb, ub).astype(np.float32)

        # ------------------------------------------------------------
        # Interior PDE points
        # X = [T, ey, epsi, vx, vy, r, delta, mu]
        # ------------------------------------------------------------
        lb_pde = [
            0.0,
            -self.ey_max,
            -self.epsi_max,
            4.0,
            -8.0,
            -self.r_max,
            -self.delta_max,
            self.mu_target_min,
        ]
        ub_pde = [
            self.Tmax,
            self.ey_max,
            self.epsi_max,
            12.0,
            8.0,
            self.r_max,
            self.delta_max,
            self.mu_target_max,
        ]
        X_pde = lhs_box(self.state_dim, nPDE, lb_pde, ub_pde)

        # ------------------------------------------------------------
        # Target boundary points around the drift-like reference
        # Unlike terminal-only survival problems, reach set is imposed
        # for all remaining times T in [0, Tmax].
        # ------------------------------------------------------------
        T = np.random.uniform(0.0, self.Tmax, size=nTarget)
        mu = np.random.uniform(self.mu_target_min, self.mu_target_max, size=nTarget)
        target = self.get_drift_target(mu)
        ey = target["ey"] + np.random.uniform(-self.ey_tol, self.ey_tol, size=nTarget)
        epsi = target["epsi"] + np.random.uniform(-self.epsi_tol, self.epsi_tol, size=nTarget)
        vx = target["vx"] + np.random.uniform(-self.vx_tol, self.vx_tol, size=nTarget)
        beta = target["beta"] + np.random.uniform(-self.beta_tol, self.beta_tol, size=nTarget)
        vy = vx * np.tan(beta)
        r = target["r"] + np.random.uniform(-self.r_tol, self.r_tol, size=nTarget)
        delta = target["delta"] + np.random.uniform(-self.delta_tol, self.delta_tol, size=nTarget)
        X_tgt = np.column_stack([T, ey, epsi, vx, vy, r, delta, mu]).astype(np.float32)

        # ------------------------------------------------------------
        # Avoid boundary points
        # Split samples among several failure modes:
        #   1) road departure ey = +/- ey_max
        #   2) excessive sideslip beta = +/- beta_max
        #   3) excessive yaw rate r = +/- r_max
        #   4) loss of forward speed vx = vx_min
        # ------------------------------------------------------------
        X_avoid = lhs_box(self.state_dim, nAvoid, lb_pde, ub_pde)
        modes = np.arange(nAvoid) % 4

        # Road departure boundary
        idx = modes == 0
        X_avoid[idx, 1] = self.ey_max * np.where(np.random.rand(np.sum(idx)) > 0.5, 1.0, -1.0)

        # Sideslip boundary: set vy from vx and beta boundary
        idx = modes == 1
        beta_bd = self.beta_max * np.where(np.random.rand(np.sum(idx)) > 0.5, 1.0, -1.0)
        X_avoid[idx, 4] = X_avoid[idx, 3] * np.tan(beta_bd)

        # Yaw-rate boundary
        idx = modes == 2
        X_avoid[idx, 5] = self.r_max * np.where(np.random.rand(np.sum(idx)) > 0.5, 1.0, -1.0)

        # Low-speed boundary
        idx = modes == 3
        X_avoid[idx, 3] = self.vx_min

        X_avoid = X_avoid.astype(np.float32)

        return (
            self.scale_state(X_pde),
            self.scale_state(X_tgt),
            self.scale_state(X_avoid),
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

    def _fiala_lateral_force_torch(self, alpha, C_alpha, Fz, mu, n):
        import torch

        D = torch.clamp(n * mu * Fz, min=1.0e-9)
        z = torch.tan(alpha)
        z_sl = 3.0 * D / C_alpha

        Fy_unsat = (
            C_alpha * z
            - (C_alpha ** 2) / (3.0 * D) * torch.abs(z) * z
            + (C_alpha ** 3) / (27.0 * D ** 2) * z ** 3
        )
        Fy_sat = D * torch.sign(alpha)
        return torch.where(torch.abs(z) < z_sl, Fy_unsat, Fy_sat)

    def evaluate_physics_model_torch(self, X_pde, U_pde):
        """
        Torch version of evaluate_physics_model.

        X_pde and U_pde are scaled tensors. Returned tensors stay on the
        input device, avoiding GPU/CPU round trips in the HJB update.
        """
        import torch

        X_center = torch.as_tensor(self.X_center, dtype=X_pde.dtype, device=X_pde.device)
        X_scale = torch.as_tensor(self.X_scale, dtype=X_pde.dtype, device=X_pde.device)
        U_center = torch.as_tensor(self.U_center, dtype=U_pde.dtype, device=U_pde.device)
        U_scale = torch.as_tensor(self.U_scale, dtype=U_pde.dtype, device=U_pde.device)

        X = X_center + X_scale * X_pde
        U = U_center + U_scale * U_pde

        delta_dot = torch.clamp(U[:, 0], -self.delta_dot_max, self.delta_dot_max)
        Fx = torch.clamp(U[:, 1], self.Fx_min, self.Fx_max)

        ey = X[:, 1]
        epsi = X[:, 2]
        vx = torch.clamp(X[:, 3], min=1.0e-3)
        vy = X[:, 4]
        r = X[:, 5]
        delta = X[:, 6]
        mu = torch.clamp(X[:, 7], self.mu_min, self.mu_max)

        alpha_f = delta - torch.atan2(vy + self.lf * r, vx)
        alpha_r = -torch.atan2(vy - self.lr * r, vx)

        Fzf = self.m * self.g * self.lr / (self.lf + self.lr)
        Fzr = self.m * self.g * self.lf / (self.lf + self.lr)

        eps = 1.0e-9
        Fx_bound = 0.999 * mu * Fzr
        Fx = torch.clamp(Fx, -Fx_bound, Fx_bound)
        nF = torch.ones_like(mu)
        nR = torch.sqrt(torch.clamp((mu * Fzr) ** 2 - Fx ** 2, min=0.0)) / (mu * Fzr + eps)

        Fyf = self._fiala_lateral_force_torch(alpha_f, self.Cf, Fzf, mu, nF)
        Fyr = self._fiala_lateral_force_torch(alpha_r, self.Cr, Fzr, mu, nR)

        f = torch.zeros_like(X)
        f[:, 0] = -1.0

        denom = torch.clamp(1.0 - self.kappa_ref * ey, min=0.2)
        s_dot = (vx * torch.cos(epsi) - vy * torch.sin(epsi)) / denom
        f[:, 1] = vx * torch.sin(epsi) + vy * torch.cos(epsi)
        f[:, 2] = r - self.kappa_ref * s_dot
        f[:, 3] = (Fx - Fyf * torch.sin(delta)) / self.m + vy * r
        f[:, 4] = (Fyr + Fyf * torch.cos(delta)) / self.m - vx * r
        f[:, 5] = (self.lf * Fyf * torch.cos(delta) - self.lr * Fyr) / self.Iz
        f[:, 6] = delta_dot
        f[:, 7] = 0.0

        sigma = torch.zeros_like(X)
        sigma[:, 0] = self.sigma_T
        sigma[:, 1] = self.sigma_ey
        sigma[:, 2] = self.sigma_epsi
        sigma[:, 3] = self.sigma_vx
        sigma[:, 4] = self.sigma_vy
        sigma[:, 5] = self.sigma_r
        sigma[:, 6] = self.sigma_delta
        sigma[:, 7] = self.sigma_mu

        return f / X_scale, sigma / X_scale, True


    ################################################
    # Evaluation states for visualization
    ################################################
    def make_eval_states(self, T, num_grid, plane="beta_r", mu=0.8):
        """
        Create evaluation grid for plotting learned values.

        plane="beta_r":
            x-axis beta, y-axis r, with other variables set to the
            interpolated drift target at the requested mu.

        plane="ey_epsi":
            x-axis ey, y-axis epsi, with other variables set to the
            interpolated drift target at the requested mu.
        """
        target = self.get_drift_target(mu)
        if plane == "beta_r":
            beta_grid = np.linspace(-self.beta_max, self.beta_max, num=num_grid)
            r_grid = np.linspace(-self.r_max, self.r_max, num=num_grid)
            BETA, R = np.meshgrid(beta_grid, r_grid, indexing="ij")

            Ts = np.full_like(BETA, T)
            EY = np.full_like(BETA, target["ey"])
            EPSI = np.full_like(BETA, target["epsi"])
            VX = np.full_like(BETA, target["vx"])
            VY = VX * np.tan(BETA)
            DELTA = np.full_like(BETA, target["delta"])
            MU = np.full_like(BETA, mu)

            state = np.stack([Ts, EY, EPSI, VX, VY, R, DELTA, MU], axis=-1).reshape(-1, self.state_dim)
            meta = {"x": beta_grid, "y": r_grid, "xlabel": "beta [rad]", "ylabel": "r [rad/s]", "T": T, "plane": plane}
            return state.astype(np.float32), meta

        elif plane == "ey_epsi":
            ey_grid = np.linspace(-self.ey_max, self.ey_max, num=num_grid)
            epsi_grid = np.linspace(self.reset_epsi_min, self.reset_epsi_max, num=num_grid)
            EY, EPSI = np.meshgrid(ey_grid, epsi_grid, indexing="ij")

            Ts = np.full_like(EY, T)
            VX = np.full_like(EY, target["vx"])
            VY = VX * np.tan(target["beta"])
            R = np.full_like(EY, target["r"])
            DELTA = np.full_like(EY, target["delta"])
            MU = np.full_like(EY, mu)

            state = np.stack([Ts, EY, EPSI, VX, VY, R, DELTA, MU], axis=-1).reshape(-1, self.state_dim)
            meta = {"x": ey_grid, "y": epsi_grid, "xlabel": "ey [m]", "ylabel": "epsi [rad]", "T": T, "plane": plane}
            return state.astype(np.float32), meta

        else:
            raise ValueError("plane must be either 'beta_r' or 'ey_epsi'.")


##########################################################
# Test env implementation
##########################################################
if __name__ == "__main__":
    
    ############################
    # Drift equilibrium
    ############################
    rlenv = Env()



    ############################
    # Common dynamics
    ############################
    # x = [
    #     1.0,              # T
    #     0.0,              # ey
    #     rlenv.epsi_d,      # epsi
    #     rlenv.vx_d,       # vx
    #     rlenv.vy_d,       # vy
    #     rlenv.r_d,        # r
    #     rlenv.delta_d,    # delta
    #     0.8,              # mu
    # ]
    # u = [0.0, 1000.0]
    # f, sigma = rlenv.drift_and_diffusion(x, u)
    # print("f:", np.round(f, 4))
    # print("sigma:", np.round(sigma, 4))

    # X = np.array([x, x, x], dtype=np.float32)
    # U = np.array([[0.0, 0.0], [0.2, 1000.0], [-0.2, -2000.0]], dtype=np.float32)
    # F, SIGMA = rlenv.drift_and_diffusion(X, U)
    # print("F:\n", np.round(F, 4))
    # print("SIGMA:\n", np.round(SIGMA, 4))
    # print("============")

    ############################
    # RL steps
    ############################
    #state = rlenv.reset()
    # print("initial:", np.round(state, 3))
    # for t in range(50):
    #     action = np.array([0.0, 500.0], dtype=np.float32)
    #     next_state, reward, done, info = rlenv.step(action)
    #     print(
    #         f"t={t+1:02d}",
    #         "state:", np.round(next_state, 3),
    #         "reward:", reward,
    #         "done:", done,
    #         "info:", info,
    #     )
    #     if done:
    #         print("terminated")
    #         break
    # print("============")

    ############################
    # PINN functions
    ############################
    # X_pde, X_tgt, X_avoid = rlenv.sample_pinn_collocation_points(nPDE=3, nTarget=2, nAvoid=4)
    # print("Xpde: [T, ey, epsi, vx, vy, r, delta, mu]\n", np.round(X_pde, 3))
    # print("Xtgt: [T, ey, epsi, vx, vy, r, delta, mu]\n", np.round(X_tgt, 3))
    # print("Xavoid: [T, ey, epsi, vx, vy, r, delta, mu]\n", np.round(X_avoid, 3))

    # U_pde = np.zeros((X_pde.shape[0], rlenv.action_dim), dtype=np.float32)
    # f, sigma, diag = rlenv.evaluate_physics_model(X_pde, U_pde)
    # print("f:\n", np.round(f, 4))
    # print("sigma:\n", np.round(sigma, 4))
    # print("diag:", diag)
    # print("============")

    ############################
    # Evaluation grid
    ############################
    # states, meta = rlenv.make_eval_states(T=1.0, num_grid=20, plane="beta_r", mu=0.8)
    # print("eval states shape:", states.shape)
    # print("meta:", meta)
