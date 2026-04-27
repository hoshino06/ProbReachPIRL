# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:50:59 2024
@author: hoshino
"""
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from agent.TD3 import PIRLAgent

################################
# Load agent    
################################
log_dir     = 'logs/0427_0623_seed_1'
log_dir     = 'logs/0427_1651_seed_1'
check_point = '1000'

agent = PIRLAgent.from_checkpoint(log_dir+'/ckpt-'+check_point)

################################
# Calculate Safe probability    
################################
num_grid = 100

x1 = np.linspace(-1.5, 1.5, num=num_grid)
x2 = np.linspace(-1.2, 1.2, num=num_grid)
X1, X2 = np.meshgrid(x1, x2, indexing='ij')
T = np.full_like(X1, 02.0)

state = np.stack([T, X1, X2], axis=-1).reshape(-1, agent.config.state_dim)
value = agent.get_value(state).reshape(num_grid, num_grid)

################################
# Plot  
################################
plt.figure(figsize=(8, 6))
cont = plt.contourf(X1, X2, value, levels=50)
plt.colorbar(cont)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

#plt.contourf(X1.numpy(), X2.numpy(), max_q_values_np, levels=50, cmap='viridis')

