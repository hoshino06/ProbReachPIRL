# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:19:41 2026
@author: hoshino
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import copy
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass, asdict

########################################################################
# Neural Networks (actor and critic)
########################################################################
def get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "softplus":
        return nn.Softplus()
    elif name in ["identity", "none", "linear"]:
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation: {name}")


def initialize_linear(layer: nn.Linear, init_type: str = "default", bias_value: float = 0.0):
    init_type = init_type.lower()

    if init_type == "default":
        return
    elif init_type in ["glorot_uniform", "xavier_uniform"]:
        nn.init.xavier_uniform_(layer.weight)
    elif init_type in ["glorot_normal", "xavier_normal"]:
        nn.init.xavier_normal_(layer.weight)
    elif init_type == "kaiming_uniform":
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    elif init_type == "kaiming_normal":
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    else:
        raise ValueError(f"Unknown init_type: {init_type}")
        
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_value)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(32, 32), 
                 hidden_activation="relu", output_activation="tanh"
                 ):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(get_activation(hidden_activation))
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(get_activation(output_activation))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)


class QNetwork(nn.Module):
    def __init__(self, input_dim,
                hidden_dims=(32, 32),
                hidden_activation="tanh",
                output_activation="sigmoid",
                init_type="xavier_uniform",
                bias_value=0.0,
                ):
        super().__init__()
        layers = []
        linear_layers = []

        in_dim = input_dim
        for h in hidden_dims:
            linear = nn.Linear(in_dim, h)
            layers.append(linear)
            layers.append(get_activation(hidden_activation))
            linear_layers.append(linear)
            in_dim = h
        linear = nn.Linear(in_dim, 1)
        layers.append(linear)
        layers.append(get_activation(output_activation))
        linear_layers.append(linear)
        self.net = nn.Sequential(*layers)

        for layer in linear_layers:
            initialize_linear(layer, init_type=init_type, bias_value=bias_value)

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=(32, 32),
        hidden_activation="relu",
        output_activation="identity",
        init_type="xavier_uniform",
        bias_value=0.0,
    ):
        super().__init__()
        input_dim = state_dim + action_dim

        self.q1_net = QNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            init_type=init_type,
            bias_value=bias_value,
        )

        self.q2_net = QNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            init_type=init_type,
            bias_value=bias_value,
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1_net(sa)
        return q1

    
###########################################################################
# Reply buffer
###########################################################################
class ReplayMemory(object):
    def __init__(self, state_dim, action_dim, max_size):
        self.max_size = max_size
        self.ptr = 0   # data pointer
        self.size = 0
        self.state      = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action     = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward     = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done   = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, next_state, reward, done):
        # buffering
        self.state[self.ptr]  = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done
        # move pointer
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.state[ind], dtype=torch.float32, device=device),
            torch.as_tensor(self.action[ind], dtype=torch.float32, device=device),
            torch.as_tensor(self.next_state[ind], dtype=torch.float32, device=device),
            torch.as_tensor(self.reward[ind], dtype=torch.float32, device=device),
            torch.as_tensor(self.not_done[ind], dtype=torch.float32, device=device),
        )

    def __len__(self):
        return self.size
    
    
###########################################################################
# PIRL agent
###########################################################################
@dataclass
class AgentConfig:
    # environment / dimensions
    state_dim: int
    action_dim: int

    # actor
    actor_hidden_dims: tuple[int, ...] = (32, 32)
    actor_hidden_activation: str = "relu"
    actor_output_activation: str = "tanh"

    # critic
    critic_hidden_dims: tuple[int, ...] = (32, 32)
    critic_hidden_activation: str = "tanh"
    critic_output_activation: str = "sigmoid"

    # optimization
    critic_lr: float = 1e-3
    actor_lr: float  = 1e-3

    # RL params
    discount: float = 1.00 # Discount factor
    replay_memory_size: int = 5000
    learn_policy_noise: float = 0.2 
    learn_noise_clip: float = 0.5
    policy_updata_freq: int = 2
    

class PIRLAgent:
    def __init__(self, config: AgentConfig, device: str = "auto"):
        # Config
        self.config = config
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.device = torch.device(device)
        self.action_dim = config.action_dim
        
        # counters
        self.itr = 0

        # Neural Networks (Actor and Critic) and Optimizer (Adam)
        self.actor = Actor(config.state_dim, config.action_dim, 
                           config.actor_hidden_dims,
                           config.actor_hidden_activation, 
                           config.actor_output_activation,
                           ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), 
                                                lr=config.actor_lr)
        self.critic = Critic(config.state_dim, config.action_dim, 
                             config.critic_hidden_dims,
                             config.critic_hidden_activation, 
                             config.critic_output_activation,
                             ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), 
                                                 lr=config.critic_lr)

        # Replay Memory
        self.replay_memory = ReplayMemory(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            max_size=config.replay_memory_size,
        )
        
    
    def get_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()

    def get_action_with_learn_policy_noise(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            # next action
            noise = (
                torch.randn(len(state), self.config.action_dim) * self.config.learn_policy_noise
                ).clamp(-self.config.learn_noise_clip, self.config.learn_noise_clip)    
            action = self.actor_target(state) + noise
        return action
        
    def get_value(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.actor(state)
            value = self.critic.Q1(state, action)
        return value.cpu().numpy()   

    def get_action_value(self, state, action):
        state  = torch.as_tensor(state,  dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            value = self.critic.Q1(state, action)
        return value.cpu().numpy()   
        
    def add_to_replay_memory(self,  state, action, next_state, reward, done):

        self.replay_memory.add(state, action, next_state, reward, done)        
    
    def sample_from_replay_memory(self, minibatch_size):

        # Sample replay buffer 
        state, action, next_state, reward, not_done = self.replay_memory.sample(
            minibatch_size, self.device,
        )
        replay_samples = {
            "state": state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "not_done": not_done,
            }

        return replay_samples

    
    ####################################################################
    # Save and Load
    ####################################################################
    def get_checkpoint(self):
        return {
            "config": asdict(self.config),
            "itr": self.itr,
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def save(self, path: str):
        torch.save(self.get_checkpoint(), path)
    
    @classmethod
    def from_checkpoint(
            cls,
            path: str,
            device: str = "auto",
            load_optimizer: bool = True,
            strict: bool = True,
            ):
        # Load checkpoint
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        checkpoint = torch.load(path, map_location=device)
        if "config" not in checkpoint:
            raise KeyError("Checkpoint does not contain config.")
        
        # build agent
        config = AgentConfig(**checkpoint["config"])
        agent  = cls(config=config, device=device)
    
        # counter
        agent.itr = checkpoint.get("itr", 0)

        # load networks
        agent.actor.load_state_dict(checkpoint["actor"], strict=strict)
        agent.actor_target.load_state_dict(checkpoint["actor_target"], strict=strict)
        agent.critic.load_state_dict(checkpoint["critic"], strict=strict)
        agent.critic_target.load_state_dict(checkpoint["critic_target"], strict=strict)
    
        # optimizer
        if load_optimizer:
            agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
    
        return agent    
        
    ####################################################################
    # Update of actor and critic
    ####################################################################    
    def calculate_TD_critic_loss(self, state, action, next_state, reward, not_done):
 
        # Calculate target
        with torch.no_grad():
            # next action
            noise = (
                torch.randn_like(action) * self.config.learn_policy_noise
                ).clamp(-self.config.learn_noise_clip, self.config.learn_noise_clip)    
            next_action = self.actor_target(next_state) + noise
        
            # target Q
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.config.discount * target_Q
    
        # current Q
        current_Q1, current_Q2 = self.critic(state, action)
     
        # critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        return critic_loss


    def calculate_HJB_loss(self, X_pde, f, sigma=None, diag=False):

        # From numpy to tensor
        X_pde = torch.as_tensor(X_pde, device=self.device)    
        X_pde.requires_grad_(True)
        f     = torch.as_tensor(f, device=self.device)
        if sigma is not None:
            sigma = torch.as_tensor(sigma, device=self.device)
        
        # Convection term
        V1, V2 = self.critic(X_pde, self.actor(X_pde))    
        dV_dx_1 = torch.autograd.grad(V1.sum(), X_pde, create_graph=True)[0]
        dV_dx_2 = torch.autograd.grad(V2.sum(), X_pde, create_graph=True)[0]    
        conv_term_1 = (dV_dx_1 * f).sum(dim=1, keepdim=True) 
        conv_term_2 = (dV_dx_2 * f).sum(dim=1, keepdim=True)

        # No diffusion case
        if sigma is None:
            residual_1 = conv_term_1
            residual_2 = conv_term_2
        
        # With diagonal sigma
        elif diag:
            if sigma.dim() != 2:
                raise ValueError("For diag=True, sigma must have shape (Ns, Xdim).")

            # Calculate diagnal terms of Laplacian
            laplace_diag_1 = torch.stack([
                torch.autograd.grad(dV_dx_1[:, i].sum(), X_pde, retain_graph=True)[0][:, i]
                for i in range(X_pde.size(1))
            ], dim=1)   # (Ns, Xdim)            
            laplace_diag_2 = torch.stack([
                torch.autograd.grad(dV_dx_2[:, i].sum(), X_pde, retain_graph=True)[0][:, i]
                for i in range(X_pde.size(1))
            ], dim=1)   # (Ns, Xdim)
            
            # Diffusion term
            diff1 = ((sigma ** 2) * laplace_diag_1).sum(dim=1, keepdim=True)  # (Ns,1)
            diff2 = ((sigma ** 2) * laplace_diag_2).sum(dim=1, keepdim=True)  # (Ns,1)
            
            residual_1 = conv_term_1 + 0.5*diff1
            residual_2 = conv_term_2 + 0.5*diff2
           
        # With full sigma  
        else:
            if sigma.dim() != 3:
                raise ValueError("For diag=False, sigma must have shape (Ns, Xdim, Nw).")

            def hvp(dV_dx, v):
                gv = (dV_dx * v).sum(dim=1)
                hv = torch.autograd.grad(gv.sum(), X_pde,
                                         create_graph=True, retain_graph=True)[0]
                return (hv * v).sum(dim=1, keepdim=True)

            vs = [sigma[:, :, k] for k in range(sigma.size(2))]
            diff1 = torch.stack([hvp(dV_dx_1, v) for v in vs], dim=0).sum(dim=0)
            diff2 = torch.stack([hvp(dV_dx_2, v) for v in vs], dim=0).sum(dim=0)
                       
            residual_1 = conv_term_1 + 0.5 * diff1
            residual_2 = conv_term_2 + 0.5 * diff2
        
        pde_loss =  (residual_1 ** 2).mean() + (residual_2 ** 2).mean()

        return pde_loss

    def calculate_boundary_loss(self, X_tgt, X_avoid):

        X_tgt   = torch.as_tensor(X_tgt,  dtype=torch.float32, device=self.device)
        X_avoid = torch.as_tensor(X_avoid,dtype=torch.float32, device=self.device)
        
        boundary_loss = 0.0

        # Target boundary
        V_ini1, V_ini2 = self.critic(X_tgt, self.actor(X_tgt))
        boundary_loss += (F.mse_loss(V_ini1, torch.ones_like(V_ini1)) 
                          + F.mse_loss(V_ini2, torch.ones_like(V_ini2)))
        # Avoid boundary
        V_lat1, V_lat2 = self.critic(X_avoid, self.actor(X_avoid))
        boundary_loss += (F.mse_loss(V_lat1, torch.zeros_like(V_lat1)) 
                          + F.mse_loss(V_lat2, torch.zeros_like(V_lat2)) )
        
        return boundary_loss

    def update_critic(
            self, weight_td3 = 1, weight_hjb = 0, weight_bdr = 0,
            replay_samples = None, 
            X_pde = None, f = None, sigma=None, diag=False, 
            X_tgt= None, X_avoid = None
            ):

        # Loss calculation        
        critic_loss = 0.0
        self.critic_optimizer.zero_grad()
        loss_dict = {"td":0, "hjb":0, "bdr": 0}

        if weight_td3 > 0:
            td3_loss = self.calculate_TD_critic_loss(**replay_samples)
            critic_loss +=  weight_td3 * td3_loss
            loss_dict["td"] = td3_loss.item()
            
        if weight_hjb > 0 and X_pde is not None:
            hjb_loss = self.calculate_HJB_loss(X_pde, f, sigma, diag)
            critic_loss += weight_hjb * hjb_loss
            loss_dict["hjb"] = hjb_loss.item()
            
        if weight_bdr > 0 and X_tgt is not None and X_avoid is not None:
            boundary_loss = self.calculate_boundary_loss(X_tgt, X_avoid)
            critic_loss +=  weight_bdr * boundary_loss
            loss_dict["bdr"] = boundary_loss.item()

        critic_loss.backward()
        self.critic_optimizer.step()

        return loss_dict

    def update_actor(self, state):

        # Compute actor losse
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_target_newtorks(self, tau = 0.005):

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


####################################################################
# Training loop
####################################################################
def train(env,  
          agent, 
          num_episodes, 
          seed      = 1,
          log_dir   = None,
          checkpoint_freq = 1000,
          verbose   = 1,
          # ---- weight config ----
          loss_weights: tuple[float, float, float] = (1, 0, 0), #(td3, hjb, bdr)
          weight_schedule: tuple[float, float, float, float] | None = None, # (center, steepness, del_start, del_end) 
          # ---- RL config ----
          initial_exploration_num = 100,
          exploration_noise       = 0.1, 
          minibatch_size          = 128,
          policy_update_freq      = 2,  
          target_update_rate      = 0.005,
          # ---- PINN config ----
          num_collocations: tuple[int, int, int] = (64, 32, 32) #(nPDE, nTGT, nAVOID)
          ):
 
    ######################################
    # Prepare log writer 
    ######################################
    if log_dir:
        summary_dir    = log_dir+'/'+datetime.now().strftime('%m%d_%H%M')+f'_seed_{seed}'
        summary_writer = SummaryWriter(log_dir=summary_dir)
        if verbose >= 1:
            print(f'Progress recorded in {summary_dir}')
            print(f'---> $tensorboard --logdir {summary_dir}')        
                
    ######################################
    # Initalizations 
    ######################################
    policy_update_cnt = 0    
    weight_delta = 1
    weight_td3, weight_hjb, weight_bdr = loss_weights
    nPDE, nTarget, nAvoid = num_collocations
    
    ##########################################
    # Training loop
    ##########################################
    start, end = (agent.itr, agent.itr+num_episodes)
    if verbose < 1:
        iterator = range(start+1, end+1)
    else:
        iterator = tqdm(range(start+1, end+1), ascii=True, unit='episodes')        
    
    for ep in iterator:
        
        ##########################
        # Reset
        ##########################
        state, is_done = (env.reset(), False)
        episode_reward = 0
        episode_q0     = agent.get_value(state) 

        ##########################
        # Weight scheduling
        ##########################
        if weight_schedule is not None:
            center      = weight_schedule[0] # ep_num half
            sharpness   = weight_schedule[1] # steepness
            delta_start = weight_schedule[2]
            delta_end   = weight_schedule[3]
            sigm = 1.0 / (1.0 + np.exp(-sharpness * (ep - center)))
            if sigm < 0.01:
                sigm = 0
            elif sigm > 0.99:
                sigm = 1
            weight_delta = delta_start + (delta_end - delta_start) *sigm 
            weight_td3 = loss_weights[0] * (1-weight_delta)  
            weight_hjb = loss_weights[1] * weight_delta
            weight_bdr = loss_weights[2] * weight_delta
        
        #######################################################
        # Iterate until episode ends 
        #######################################################
        while not is_done:
            
            ####################
            # Get action
            ####################
            # Initial random exploration
            if ep  <= initial_exploration_num:
                act_fn = agent.config.actor_output_activation          
                if act_fn == "tanh":
                    # [-max_action, max_action]
                    action = np.random.rand(agent.action_dim) * 2 - 1                
                elif act_fn == "sigmoid":
                    # [0, max_action]
                    action = np.random.rand(agent.action_dim)
                else:
                    raise ValueError(f"Unsupported actor_output_activation: {act_fn}")

            # Policy-based exploration
            else:
                act_fn = agent.config.actor_output_activation
                noise  = np.random.normal(0, exploration_noise, size=agent.action_dim)
                action = agent.get_action(state) + noise
                if act_fn == "tanh":
                    action =  action.clip(-1, 1)              
                elif act_fn == "sigmoid":
                    action = action.clip(0,1)
                else:
                    raise ValueError(f"Unsupported actor_output_activation: {act_fn}")

            ###############################
            # Make a step (Rollout)
            ###############################
            # Env step
            next_state, reward, is_done, info = env.step(action)
            episode_reward += reward
            
            # Add to replay memory
            agent.add_to_replay_memory(state, action, next_state, reward, is_done)
            
            # Update state
            state = next_state

            ###############################
            # Learn
            ###############################
            if ep > initial_exploration_num: 
                
                # Sample from replay memory
                replay_samples = agent.sample_from_replay_memory(minibatch_size)
                
                # Sample for PINN update 
                X_pde, X_tgt, X_avoid = env.sample_pinn_collocation_points(nPDE=nPDE, 
                                                                           nTarget=nTarget, 
                                                                           nAvoid=nAvoid
                                                                           )
                #U_pde = agent.get_action(X_pde)
                U_pde = agent.get_action_with_learn_policy_noise(X_pde)

                f, sigma, diag  = env.evaluate_physics_model(X_pde, U_pde)

                # Update critic
                loss = agent.update_critic(weight_td3 = weight_td3, 
                                            weight_hjb = weight_hjb,  
                                            weight_bdr = weight_bdr,
                                            replay_samples = replay_samples, 
                                            X_pde = X_pde, f = f, sigma=sigma, diag=diag,
                                            X_tgt = X_tgt, X_avoid = X_avoid
                                            )
                     
                # Delayed policy updates
                policy_update_cnt = (policy_update_cnt+1) % policy_update_freq
                if policy_update_cnt == 0:
                    
                    # Update actor
                    agent.update_actor(replay_samples["state"])

                    # Update target networks
                    agent.update_target_newtorks(tau = target_update_rate) 
            else:
                loss = {"td":0, "hjb":0, "bdr": 0}
        
        #############################
        # Episode cleanup        
        #############################
        # Update agent counter
        agent.itr = ep
        
        # Evaluate Loss terms (if needed)
        if loss["td"] == 0:
            replay_samples  = agent.sample_from_replay_memory(minibatch_size)
            loss["td"] = agent.calculate_TD_critic_loss(**replay_samples)
        if loss["hjb"] == 0 or loss["bdr"] == 0:
            X_pde, X_safe, X_lat = env.sample_pinn_collocation_points(nPDE=nPDE, 
                                                                      nTarget=nTarget, 
                                                                      nAvoid=nAvoid
                                                                      )
            U_pde = agent.get_action(X_pde)
            f, sigma, diag   = env.evaluate_physics_model(X_pde, U_pde)
            loss["hjb"] = agent.calculate_HJB_loss(X_pde, f, sigma, diag) 
            loss["bdr"] = agent.calculate_boundary_loss(X_safe, X_lat)
            
        
        ###################################################
        # Log
        ###################################################
        if log_dir is not None: 
            summary_writer.add_scalar("RL/Episode Reward", episode_reward, ep)
            summary_writer.add_scalar("RL/Episode Q0",     episode_q0,     ep)
            summary_writer.add_scalar("Loss/RL",  loss["td"],  ep)
            summary_writer.add_scalar("Loss/HJB", loss["hjb"], ep)
            summary_writer.add_scalar("Loss/BDR", loss["bdr"], ep)
            summary_writer.add_scalar("Weights/RL",  weight_td3, ep)
            summary_writer.add_scalar("Weights/HJB", weight_hjb, ep)
            summary_writer.add_scalar("Weights/BDR", weight_bdr, ep)
            

            summary_writer.flush()
            
            if ep % checkpoint_freq == 0 or ep == end:                
                agent.save(summary_dir+f'/ckpt-{ep}')




##########################################################
# Test of PDE loss calculation
##########################################################
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    config = AgentConfig(
        state_dim=4,
        action_dim=1,
    )
    agent = PIRLAgent(config, device)

    ##################################
    # HJB loss calculation 
    ##################################    
    #---- Diagonal case ----
    Ns = 5 # num of samples
    Xdim = 4 # state dim
    X_pde = np.random.randn(Ns, Xdim).astype(np.float32)
    f     = np.random.randn(Ns, Xdim).astype(np.float32)
    
    sigma_diag = np.zeros((Ns, Xdim), dtype=np.float32)
    sigma_diag[:, :3] = 0.1  # 最初の3成分非ゼロ
        
    t0 = time.perf_counter()
    for i in range(10):
        loss = agent._calculate_HJB_loss(X_pde, f, sigma_diag, diag=True)
    t1 = time.perf_counter()
    print("time (diag): ", t1-t0)

    #--- Full sigma ---
    Ns = 5
    Xdim = 4
    Nw = 4
    
    sigma_full = np.zeros((Ns, Xdim, Nw), dtype=np.float32)
    sigma_full[:, 0, 0] = 0.1
    sigma_full[:, 0, 1] = 0.05
    sigma_full[:, 1, 1] = 0.1 #0.02
    sigma_full[:, 2, 2] = 0.1 #0.02
    sigma_full[:, 3, 3] = 0.1 #0.02
     
    t0 = time.perf_counter()
    for i in range(10):
        loss_full = agent._calculate_HJB_loss(X_pde, f, sigma_full, diag=False)
    t1 = time.perf_counter()    
    print("time (full): ", t1-t0)

