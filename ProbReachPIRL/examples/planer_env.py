# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:25:12 2026
@author: hoshino
"""
import numpy as np
from scipy.stats import qmc

class Env(object):
    def __init__(self):        
        self.dt = 0.1;
        self.state_dim  = 3
        self.action_dim = 1
        
    ################################################
    # Common dynamics model
    ################################################
    def drift_and_diffusion(self, X, U):
        """
        Continuous-time dynamics:
            dX = f(X, U) dt + sigma(X,U) dW 

        X: shape (X_dim, ) or (N, X_dim)
        U: shape (U_dim, ) or (N, U_dim)
        """
        X = np.asarray(X, dtype=np.float32)
        U = np.asarray(U, dtype=np.float32)

        single = (X.ndim == 1)
        if single:
            X = X[None, :]
            U = U[None, :]

        X1 = X[:, 1]
        X2 = X[:, 2]
        U  = U[:, 0]

        ######################
        # Drift 
        ######################
        f = np.zeros_like(X, dtype=np.float32)
        # Dynamics of the remaing time
        f[:, 0] =  - 1.0       
        # dX1/dt
        f[:, 1] = - X1**3 - X2
        # dX2/dt
        f[:, 2] =   X1 + X2 + U

        ######################
        # Diffusion
        ######################
        Ns = X.shape[0]
        Xdim = X.shape[1]
        sigma = np.zeros((Ns, Xdim), dtype=np.float32)             
        sigma[:, 1] = 1.0
        sigma[:, 2] = 1.0
        
        return (f[0], sigma[0]) if single else (f, sigma)
    
        
    ################################################
    # for RL training
    #################################################    
    def reset(self):
        T   = 2.0
        s   = np.sign( np.random.randn(2) )
        r   = np.random.rand(2)
        X1  = s[0]*r[0]*1.5
        X2  = s[1]*r[1]*1.2
        self.state = np.array([T, X1, X2])
        return self.state

    def step(self, action):

        # Check Terminal condition
        T  = self.state[0]
        X2 = self.state[2]        
        
        isTimeOver = (T < self.dt)
        isUnsafe   = abs( X2 ) > 1
        done       = isTimeOver or isUnsafe
        
        # Calculate next state by Euler-Maruyama scheme
        action   = np.asarray(action, dtype=np.float32).reshape(-1)
        f, sigma = self.drift_and_diffusion(self.state, action)
        noise    = np.random.randn(self.state_dim).astype(np.float32)        
        next_state = self.state + self.dt * f + np.sqrt(self.dt) * sigma * noise
        self.state = next_state

        # Reward
        if done and (not isUnsafe):
            reward = 1.0
        else:
            reward = 0
 
        info = {}
        
        return next_state, reward, done, info

    ################################################
    # for PINN training 
    #################################################    
    def sample_pinn_collocation_points(self, nPDE, nSAFE, nBDR):

        def lhs_box(dim, num, lb, ub):
            u = qmc.LatinHypercube(d=dim).random(num)
            return qmc.scale(u, lb, ub).astype(np.float32)

        # Interior points    
        X_pde = lhs_box(3, nPDE, [0, -1.5, -1.0], [2.0, 1.5, 1.0])
        
        # Safe and Terminal boundary (at T=0)
        X = lhs_box(2, nSAFE, [-1.5, -1.0], [1.5, 1.0])
        T = np.zeros(nSAFE)
        X_safe = np.column_stack([T, X]).astype(np.float32)
        
        # Unsafe Lateral boundary (unsafe set)        
        T_X1 = lhs_box(2, nBDR, [0.0, -1.5], [2.0, 1.5])
        X2 = np.ones(nBDR)
        X2[nBDR//2:] = -1.0
        X_lat = np.column_stack([T_X1, X2]).astype(np.float32)

        return X_pde, X_safe, X_lat       
        
        
    def evaluate_physics_model(self, X_pde, U_pde):

        f, sigma = self.drift_and_diffusion(X_pde, U_pde)
        diag     = True  
    
        return f, sigma, diag
           




##########################################################
# Test env implementation
##########################################################
if __name__ == "__main__":
    
    
    rlenv = Env()

    ############################
    # Common dynamics
    ############################
    # Drift term
    x = [0.5, 0.5, 10]
    u = [1]
    f, sigma = rlenv.drift_and_diffusion(x, u)
    print('f: ', f)
    print('sigma: ', sigma)
    
    X = [[0.5, 0.5, 10], [0.3,0.3, 10], [-0.1,0.1, 10]]
    U = [[1], [-0.1], [0.1]]
    F, SIGMA = rlenv.drift_and_diffusion(X, U)   
    print('F: ', F)
    print('SIGMA: ', SIGMA)
    print('============')
    
    ############################
    # RL steps
    ############################
    state = rlenv.reset()
    print("initial:", np.round(state, 2))
    for t in range(20):
        next_state, reward, done, info = rlenv.step([0])
        print(f"t={t+1} state:", np.round(state, 2), "  reward:", reward, "done:", done)
        state = next_state
        if done:
            print("terminated")
            break
    print('============')    

    ############################
    # PINN functions
    ############################       
    X_pde,X_safe,X_lat = rlenv.sample_pinn_collocation_points(nPDE=3, nSAFE=2, nBDR=4)
    print('Xpde: [T, X1, X2]\n', np.round(X_pde,2))
    print('Xsafe: [T, X1, X2]\n', np.round(X_safe,2))
    print('Xlat: [T, X1, X2]\n', np.round(X_lat,2))
    U_pde = np.ones((X_pde.shape[0], 1), dtype=np.float32)
    f, sigma, diag = rlenv.evaluate_physics_model(X_pde, U_pde)
    print('f: ', f)
    print('sigma: ', sigma)
        
        
        
        