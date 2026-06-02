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
import numpy as np
from scipy.stats import qmc
from scipy.optimize import least_squares

class Env(object):
    def __init__(self):
        # ------------------------------------------------------------
        # Basic dimensions
        # ------------------------------------------------------------
        self.dt = 0.02
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

        # ------------------------------------------------------------
        # Drift equilibrium target
        # ------------------------------------------------------------
        # We specify vx_d, mu_d, and kappa_d.
        # The remaining quantities ey_d(=0), epsi_d, vy_d, r_d,
        # delta_d, and Fx_d are obtained by solving the steady-state
        # equations of the path-relative dynamic bicycle model.
        # Curvature chosen so that the path-relative equilibrium is
        # consistent with the paper's drift point:
        # beta ~= -20.44 deg, r ~= 0.600 rad/s, Ux = 8 m/s.
        self.kappa_ref = 0.600 * np.cos(np.deg2rad(-20.44)) / 8.0
        self.set_drift_equilibrium(
            vx_d = 8.0,
            mu_d = 0.55,
            verbose = 2, 
        )

        # Target tolerances
        self.ey_tol = 0.25
        self.epsi_tol = np.deg2rad(8.0)
        self.vx_tol = 2.0
        self.beta_tol = np.deg2rad(8.0)
        self.r_tol = 0.35
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
        beta = self.beta(X)

        mask = (
            (np.abs(ey - self.ey_d) <= self.ey_tol)
            & (np.abs(epsi - self.epsi_d) <= self.epsi_tol)
            & (np.abs(vx - self.vx_d) <= self.vx_tol)
            & (np.abs(beta - self.beta_d) <= self.beta_tol)
            & (np.abs(r - self.r_d) <= self.r_tol)
            & (np.abs(delta - self.delta_d) <= self.delta_tol)
            & (np.abs(X[:, 7] - self.mu_d) <= 0.20)
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


    ################################################
    # Methods for RL training
    ################################################
    def reset(self):
        """Sample an initial state near, but not exactly at, the drift-like target."""
        T = self.Tmax #np.random.uniform(0.2, self.Tmax)

        # ey = np.random.uniform(-1.0, 1.0)
        # epsi = self.epsi_d + np.random.uniform(-0.35, 0.35)
        # vx = np.random.uniform(6.0, 10.0)
        # beta0 = self.beta_d + np.random.uniform(-0.35, 0.35)
        # vy = vx * np.tan(beta0)
        # r = self.r_d + np.random.uniform(-0.9, 0.9)
        # delta = self.delta_d + np.random.uniform(-0.25, 0.25)
        # mu = np.clip(self.mu_d + np.random.uniform(-0.15, 0.15), self.mu_min, self.mu_max)

        ey = np.random.uniform(-0.8, 0.8)
        epsi = self.epsi_d + np.random.uniform(-0.01, 0.01)
        vx = np.random.uniform(7.9, 8.1)
        beta0 = self.beta_d + np.random.uniform(-0.2, 0.2)
        vy = vx * np.tan(beta0)
        r = self.r_d + np.random.uniform(-0.2, 0.2)
        delta = self.delta_d + np.random.uniform(-0.01, 0.01)
        mu = np.clip(self.mu_d + np.random.uniform(-0.001, 0.001), self.mu_min, self.mu_max)
        
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
            0.35,
        ]
        ub_pde = [
            self.Tmax,
            self.ey_max,
            self.epsi_max,
            12.0,
            8.0,
            self.r_max,
            self.delta_max,
            0.85,
        ]
        X_pde = lhs_box(self.state_dim, nPDE, lb_pde, ub_pde)

        # ------------------------------------------------------------
        # Target boundary points around the drift-like reference
        # Unlike terminal-only survival problems, reach set is imposed
        # for all remaining times T in [0, Tmax].
        # ------------------------------------------------------------
        T = np.random.uniform(0.0, self.Tmax, size=nTarget)
        ey = np.random.uniform(-self.ey_tol, self.ey_tol, size=nTarget)
        epsi = np.random.uniform(self.epsi_d - self.epsi_tol, self.epsi_d + self.epsi_tol, size=nTarget)
        vx = np.random.uniform(self.vx_d - self.vx_tol, self.vx_d + self.vx_tol, size=nTarget)
        beta = np.random.uniform(self.beta_d - self.beta_tol, self.beta_d + self.beta_tol, size=nTarget)
        vy = vx * np.tan(beta)
        r = np.random.uniform(self.r_d - self.r_tol, self.r_d + self.r_tol, size=nTarget)
        delta = np.random.uniform(self.delta_d - self.delta_tol, self.delta_d + self.delta_tol, size=nTarget)
        mu = np.random.uniform(max(self.mu_min, self.mu_d - 0.15), min(self.mu_max, self.mu_d + 0.15), size=nTarget)
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


    ################################################
    # Evaluation states for visualization
    ################################################
    def make_eval_states(self, T, num_grid, plane="beta_r", mu=0.8):
        """
        Create evaluation grid for plotting learned values.

        plane="beta_r":
            x-axis beta, y-axis r, with ey=epsi=0, vx=vx_d, delta=delta_d.

        plane="ey_epsi":
            x-axis ey, y-axis epsi, with vx=vx_d, beta=beta_d, r=r_d.
        """
        if plane == "beta_r":
            beta_grid = np.linspace(-self.beta_max, self.beta_max, num=num_grid)
            r_grid = np.linspace(-self.r_max, self.r_max, num=num_grid)
            BETA, R = np.meshgrid(beta_grid, r_grid, indexing="ij")

            Ts = np.full_like(BETA, T)
            EY = np.zeros_like(BETA)
            EPSI = np.full_like(BETA, self.epsi_d)
            VX = np.full_like(BETA, self.vx_d)
            VY = VX * np.tan(BETA)
            DELTA = np.full_like(BETA, self.delta_d)
            MU = np.full_like(BETA, mu)

            state = np.stack([Ts, EY, EPSI, VX, VY, R, DELTA, MU], axis=-1).reshape(-1, self.state_dim)
            meta = {"x": beta_grid, "y": r_grid, "xlabel": "beta [rad]", "ylabel": "r [rad/s]", "T": T, "plane": plane}
            return state.astype(np.float32), meta

        elif plane == "ey_epsi":
            ey_grid = np.linspace(-self.ey_max, self.ey_max, num=num_grid)
            epsi_grid = np.linspace(-self.epsi_max, self.epsi_max, num=num_grid)
            EY, EPSI = np.meshgrid(ey_grid, epsi_grid, indexing="ij")

            Ts = np.full_like(EY, T)
            VX = np.full_like(EY, self.vx_d)
            VY = VX * np.tan(self.beta_d)
            R = np.full_like(EY, self.r_d)
            DELTA = np.full_like(EY, self.delta_d)
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
