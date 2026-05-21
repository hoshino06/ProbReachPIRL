# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:50:59 2024
@author: hoshino
"""
import numpy as np
import matplotlib.pyplot as plt
from agent.TD3 import PIRLAgent

################################
# Load agent    
################################
#log_dir     = 'logs/TD3_5000_outside_seed_1'
#log_dir     = 'logs/0430_2053_seed_1'
#check_point = '14000'

log_dir = 'logs/1D/0509_0555_seed_1'
log_dir = 'logs/1D/0508_0739_seed_1'
check_point = '20000'

agent = PIRLAgent.from_checkpoint(log_dir+'/ckpt-'+check_point)


################################
# Env 
################################
from examples.env_1d_reach import Env

################################
# Calculate Safe probability    
################################
num_grid = 100
T = 1.0
#x1 = np.linspace(-1.5, 1.5, num=num_grid)
#x2 = np.linspace(-1.2, 1.2, num=num_grid)
#X1, X2 = np.meshgrid(x1, x2, indexing='ij')
#T = np.full_like(X1, 2.0)

state, meta = Env.make_eval_states(T, num_grid)
#state = np.stack([T, X1, X2], axis=-1).reshape(-1, agent.config.state_dim)

value = agent.get_value(state) #.reshape(num_grid, num_grid)

Env.plot_values(value, meta)




import matplotlib.pyplot as plt
from matplotlib import cm
num_grid = 50
x = np.linspace(-1, 1, num_grid)
u = np.linspace(-1, 1, num_grid)
x_grid, u_grid = np.meshgrid(x, u, indexing="ij")
T_grid = np.full_like(x_grid, 2)
states = np.stack([T_grid, x_grid], axis=-1).reshape(-1, 2)
actions = u_grid.reshape(-1, 1)
values = agent.get_action_value(states, actions)
values = values.reshape(num_grid, num_grid)


# Plot action for 1D example
# fig = plt.figure(figsize=(7,5))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(
#     u_grid, x_grid, values,
#     cmap=cm.viridis,   # ← 色付け
#     edgecolor='none'
# )
# fig.colorbar(surf, ax=ax, shrink=0.7, label="Q(T=1, x, u)")
# ax.set_xlabel("u")
# ax.set_ylabel("x")
# ax.set_zlabel("Q")
# plt.show()


################################
# Plot  
################################


# plt.figure(figsize=(8, 6))
# cont = plt.contourf(X1, X2, value, levels=50)
# plt.colorbar(cont)
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.show()

#plt.contourf(X1.numpy(), X2.numpy(), max_q_values_np, levels=50, cmap='viridis')

