# -*- coding: utf-8 -*-
import ray
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import copy
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import deque

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

    def add_batch(self, states, actions, next_states, rewards, dones):
        states      = np.asarray(states, dtype=np.float32)
        actions     = np.asarray(actions, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        rewards     = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
        dones       = np.asarray(dones, dtype=np.float32).reshape(-1, 1)

        n = len(states)
        idx = (np.arange(n) + self.ptr) % self.max_size

        self.state[idx]      = states
        self.action[idx]     = actions
        self.next_state[idx] = next_states
        self.reward[idx]     = rewards
        self.not_done[idx]   = 1.0 - dones

        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

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
    replay_memory_size: int = 5_000
    learn_policy_noise: float = 0.2  #0.2 
    learn_noise_clip: float   = 0.5    
    hjb_laplacian_mode: str = "loop"

class PIRLAgent:
    def __init__(self, config:AgentConfig, device:str="auto", learner=True):
        # Config
        self.config = config
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.device = torch.device(device)
        self.action_dim = config.action_dim
        
        # counters
        self.itr = 0

        # Neural Networks (Actor and Critic)
        self.actor = Actor(config.state_dim, config.action_dim, 
                           config.actor_hidden_dims,
                           config.actor_hidden_activation, 
                           config.actor_output_activation,
                           ).to(self.device)
        self.critic = Critic(config.state_dim, config.action_dim, 
                             config.critic_hidden_dims,
                             config.critic_hidden_activation, 
                             config.critic_output_activation,
                             ).to(self.device)

        # Optimizer, Target Networks, Replay Memory (for Learner)       
        if learner:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), 
                                                    lr=config.actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), 
                                                     lr=config.critic_lr)
            self.critic_target = copy.deepcopy(self.critic)
            self.actor_target  = copy.deepcopy(self.actor)
            self.replay_memory = ReplayMemory(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                max_size=config.replay_memory_size,
            )        
            
    ####################################################################
    # Getter of actions and values
    ####################################################################    
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
                torch.randn(
                    len(state), self.config.action_dim, device=self.device
                    ) * self.config.learn_policy_noise
                ).clamp(-self.config.learn_noise_clip, self.config.learn_noise_clip)    
            action = self.actor_target(state) + noise
        return action

    def get_random_action(self): 
        act_fn = self.config.actor_output_activation
        if act_fn == "tanh":
            action = np.random.uniform(-1, 1, size=self.action_dim)
        elif act_fn == "sigmoid":
            action = np.random.uniform(0, 1, size=self.action_dim)
        else:
            raise ValueError(f"Unsupported actor_output_activation: {act_fn}")
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
        
    ####################################################################
    # Relpay memory
    ####################################################################    
    def add_to_replay_memory(self,  state, action, next_state, reward, done):
        self.replay_memory.add(state, action, next_state, reward, done)        

    def add_batch_to_replay_memory(self, transitions):
        states, actions, next_states, rewards, dones = transitions
        self.replay_memory.add_batch(states, actions, next_states, rewards, dones)    

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

    def get_checkpoint_cpu(self):
        def to_cpu(obj):
            if torch.is_tensor(obj):
                return obj.detach().cpu().clone()
            if isinstance(obj, dict):
                return {k: to_cpu(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_cpu(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(to_cpu(v) for v in obj)
            return obj

        return to_cpu(self.get_checkpoint())

    def save(self, path: str):
        torch.save(self.get_checkpoint(), path)
    
    @classmethod
    def from_checkpoint(
            cls,
            path: str,
            device: str = "auto",
            learner: bool = True,
            strict:  bool = True,
            ):
        # Load checkpoint
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        checkpoint = torch.load(path, map_location=device)
        if "config" not in checkpoint:
            raise KeyError("Checkpoint does not contain config.")
        
        # build agent
        config = AgentConfig(**checkpoint["config"])
        agent  = cls(config=config, device=device, learner=learner)
    
        # counter
        agent.itr = checkpoint.get("itr", 0)

        # load networks
        agent.actor.load_state_dict(checkpoint["actor"], strict=strict)
        agent.critic.load_state_dict(checkpoint["critic"], strict=strict)
    
        # optimizer
        if learner:
            agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            agent.actor_target.load_state_dict(checkpoint["actor_target"], strict=strict)
            agent.critic_target.load_state_dict(checkpoint["critic_target"], strict=strict)

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

    def calculate_HJB_loss(self, pde_samples, f, sigma=None, diag=False):

        # From numpy to tensor
        X_pde, U_pde = pde_samples
        X_pde = torch.as_tensor(X_pde, dtype=torch.float32, device=self.device)    
        U_pde = torch.as_tensor(U_pde, dtype=torch.float32, device=self.device)    
        X_pde.requires_grad_(True)
        f     = torch.as_tensor(f, dtype=torch.float32, device=self.device)
        if sigma is not None:
            sigma = torch.as_tensor(sigma, dtype=torch.float32, device=self.device)
        
        # Convection term
        V1, V2 = self.critic(X_pde, U_pde)
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
            laplace_mode = getattr(self.config, "hjb_laplacian_mode", "loop")
            if laplace_mode == "loop":
                laplace_diag_1 = torch.stack([
                    torch.autograd.grad(dV_dx_1[:, i].sum(), X_pde, retain_graph=True)[0][:, i]
                    for i in range(X_pde.size(1))
                ], dim=1)   # (Ns, Xdim)
                laplace_diag_2 = torch.stack([
                    torch.autograd.grad(dV_dx_2[:, i].sum(), X_pde, retain_graph=True)[0][:, i]
                    for i in range(X_pde.size(1))
                ], dim=1)   # (Ns, Xdim)
            elif laplace_mode == "batched":
                laplace_diag_1 = self._calculate_laplace_diag_batched(dV_dx_1, X_pde)
                laplace_diag_2 = self._calculate_laplace_diag_batched(dV_dx_2, X_pde)
            else:
                raise ValueError(
                    "Unsupported hjb_laplacian_mode: "
                    f"{laplace_mode!r}. Use 'loop' or 'batched'."
                )
            
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

    def _calculate_laplace_diag_batched(self, dV_dx, X):
        """Calculate per-sample Hessian diagonal using batched VJP."""
        n, d = X.shape
        eye = torch.eye(d, dtype=X.dtype, device=X.device)
        grad_outputs = eye[:, None, :].expand(d, n, d)
        hess_rows = torch.autograd.grad(
            dV_dx,
            X,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            is_grads_batched=True,
        )[0]
        return hess_rows.diagonal(dim1=0, dim2=2)

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
            pde_samples = None, f = None, sigma=None, diag=False, 
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
            
        if weight_hjb > 0 and pde_samples is not None:
            hjb_loss = self.calculate_HJB_loss(pde_samples, f, sigma, diag)
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

    def update_target_networks(self, tau = 0.005):

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


####################################################################
# Trainer
####################################################################
@ray.remote
class RolloutWorker:
    def __init__(self, worker_id, env_cls, agent_config, exploration_noise=0.1):
        self.worker_id = worker_id
        self.env = env_cls()
        self.agent = PIRLAgent(agent_config, device="cpu", learner=False)
        self.exploration_noise = exploration_noise

    def rollout(self, actor_weights_ref=None, critic_weights=None, random_action=False):

        # Initializations
        state = self.env.reset()
        done = False
        episode_reward = 0.0
        episode_q0     = np.nan
        states, actions, next_states, rewards, dones = [], [], [], [], []

        # Load network weights
        if actor_weights_ref is not None:
            self.agent.actor.load_state_dict(actor_weights_ref)
        if critic_weights is not None:
            self.agent.critic.load_state_dict(critic_weights)
            episode_q0 = self.agent.get_value(state) 
        
        # Rollout
        while not done:            
            if random_action:
                action = self.agent.get_random_action()
            else:
                action = self.agent.get_action(state)
                action = action + np.random.normal(0, self.exploration_noise,
                                                   size=self.agent.action_dim) 

            next_state, reward, done, info = self.env.step(action)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)

            episode_reward += reward
            state = next_state

        transitions = (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )
        logs = {
            "episode_reward": float(episode_reward),
            "episode_q0": float(np.asarray(episode_q0).mean()),
            }
            
        return self.worker_id, transitions, logs

@ray.remote
class Learner:
    def __init__(self, env_cls, agent_config, device="cuda", verbose=0):
        self.env = env_cls()
        self.agent = PIRLAgent(agent_config, device=device)
        self.policy_update_cnt = 0
        if verbose >= 1:
            print(
                "Learner device:",
                self.agent.device,
                "cuda_available:",
                torch.cuda.is_available(),
                "CUDA_VISIBLE_DEVICES:",
                os.environ.get("CUDA_VISIBLE_DEVICES"),
            )

    def load_checkpoint_state(self, checkpoint, strict=True):
        self.agent.itr = checkpoint.get("itr", 0)
        self.agent.actor.load_state_dict(checkpoint["actor"], strict=strict)
        self.agent.critic.load_state_dict(checkpoint["critic"], strict=strict)
        self.agent.actor_target.load_state_dict(checkpoint["actor_target"], strict=strict)
        self.agent.critic_target.load_state_dict(checkpoint["critic_target"], strict=strict)
        self.agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        return self.agent.itr

    def add_memory(self, transitions):
        self.agent.add_batch_to_replay_memory(transitions)
        return len(self.agent.replay_memory)

    def get_network_weights(self):
        actor_weights = {
            k: v.detach().cpu().clone() #From GPU to CPU
            for k, v in self.agent.actor.state_dict().items()
        }
        critic_weights = {
            k: v.detach().cpu().clone()
            for k, v in self.agent.critic.state_dict().items()
        }
        return actor_weights, critic_weights

    def get_actor_weights(self):
        return {
            k: v.detach().cpu()  #From GPU to CPU
            for k, v in self.agent.actor.state_dict().items()
        }

    def learn_once(
        self,
        minibatch_size=128,
        policy_update_freq=2,
        target_update_rate=0.005,
        loss_weights=(1.0, 0.0, 0.0),
        num_collocations=(64, 32, 32),
    ):

        weight_td3, weight_hjb, weight_bdr = loss_weights
        nPDE, nTarget, nAvoid = num_collocations

        # Sample from replay memory
        replay_samples = self.agent.sample_from_replay_memory(minibatch_size)

        # Sample for PINN update 
        X_pde, X_tgt, X_avoid = self.env.sample_pinn_collocation_points(
            nPDE=nPDE,
            nTarget=nTarget,
            nAvoid=nAvoid,
        )
        X_pde_tensor = torch.as_tensor(X_pde, dtype=torch.float32, device=self.agent.device)
        U_pde = self.agent.get_action_with_learn_policy_noise(X_pde_tensor)
        if hasattr(self.env, "evaluate_physics_model_torch"):
            with torch.no_grad():
                f, sigma, diag = self.env.evaluate_physics_model_torch(X_pde_tensor, U_pde)
        else:
            U_pde_phys = U_pde.detach().cpu().numpy() if torch.is_tensor(U_pde) else U_pde
            f, sigma, diag = self.env.evaluate_physics_model(X_pde, U_pde_phys)

        # Update critic
        loss = self.agent.update_critic(
            weight_td3=weight_td3,
            weight_hjb=weight_hjb,
            weight_bdr=weight_bdr,
            replay_samples=replay_samples,
            pde_samples= (X_pde_tensor, U_pde),
            f=f,
            sigma=sigma,
            diag=diag,
            X_tgt=X_tgt,
            X_avoid=X_avoid,
        )

        # Delayed policy and target updates
        self.policy_update_cnt = (self.policy_update_cnt + 1) % policy_update_freq
        if self.policy_update_cnt == 0:
            # Update actor
            self.agent.update_actor(replay_samples["state"])
            
            # Update target networks
            self.agent.update_target_networks(tau=target_update_rate)
            
        self.agent.itr += 1

        return loss 

    def evaluate_hjb_bdr_loss(self, num_collocations=(64, 32, 32)):
        nPDE, nTarget, nAvoid = num_collocations
        X_pde, X_tgt, X_avoid = self.env.sample_pinn_collocation_points(
            nPDE=nPDE,
            nTarget=nTarget,
            nAvoid=nAvoid,
        )
        X_pde_tensor = torch.as_tensor(X_pde, dtype=torch.float32, device=self.agent.device)
        U_pde = self.agent.get_action_with_learn_policy_noise(X_pde_tensor)
        if hasattr(self.env, "evaluate_physics_model_torch"):
            with torch.no_grad():
                f, sigma, diag = self.env.evaluate_physics_model_torch(X_pde_tensor, U_pde)
        else:
            U_pde_phys = U_pde.detach().cpu().numpy() if torch.is_tensor(U_pde) else U_pde
            f, sigma, diag = self.env.evaluate_physics_model(X_pde, U_pde_phys)
        hjb_loss = self.agent.calculate_HJB_loss((X_pde_tensor, U_pde), f, sigma, diag) 
        bdr_loss = self.agent.calculate_boundary_loss(X_tgt, X_avoid)     
        
        return (
            float(hjb_loss.detach().cpu().item()),
            float(bdr_loss.detach().cpu().item()),
        )
    
    def evaluate_td_loss(self, minibatch_size=128):
        replay_samples = self.agent.sample_from_replay_memory(minibatch_size)
        td_loss = self.agent.calculate_TD_critic_loss(**replay_samples)

        return float(td_loss.detach().cpu().item())
    
    def save(self, path):
        self.agent.save(path)
    



####################################################################
# Training loop
####################################################################
def train_distributed(
    env_cls,
    agent,
    num_iterations,
    # ---- train config -------
    seed=1,
    num_workers=4,
    log_dir   = None,
    log_freq  = 100,
    checkpoint_freq = 1000,
    verbose = 1,
    device  = 'auto',
    learner_num_gpus = None,
    log_tag = None,
    # ---- weight config ----    
    loss_weights=(1.0, 0.0, 0.0),
    weight_schedule=None,
    weight_schedule_time_base="global",
    # ---- PINN config ----
    num_collocations: tuple[int, int, int] = (64, 32, 32), #(nPDE, nTGT, nAVOID)
    # ---- RL config ----
    initial_exploration_num=100,
    initial_exploration_policy="random",
    exploration_noise=0.1,
    minibatch_size=128,
    policy_update_freq=2,
    target_update_rate=0.005,    
):

    ######################################
    # Prepare log writer 
    ######################################
    if log_dir:
        run_name = datetime.now().strftime('%m%d_%H%M')
        if log_tag:
            run_name += f'_{log_tag}'
        run_name += f'_seed_{seed}'
        summary_dir = log_dir + '/' + run_name
        summary_writer = SummaryWriter(log_dir=summary_dir)
        if verbose >= 1:
            print(f'Progress recorded in {summary_dir}')
            print(f'---> $tensorboard --logdir {summary_dir}')        
                             
    ######################################
    # Initalizations 
    ######################################  
    try: 

        ray.init(ignore_reinit_error=True, include_dashboard=False)        
        workers = [RolloutWorker.remote(i, env_cls, agent.config,  
                                        exploration_noise=exploration_noise)
                   for i in range(num_workers)
                   ]
        learner_cls = Learner
        if learner_num_gpus is not None:
            learner_cls = Learner.options(num_gpus=learner_num_gpus)
        learner = learner_cls.remote(env_cls, agent.config, device=device, verbose=verbose)
        ray.get(learner.load_checkpoint_state.remote(agent.get_checkpoint_cpu()))
        actor_weights, critic_weights = ray.get(learner.get_network_weights.remote())
        actor_weights_ref = ray.put(actor_weights)
        critic_weights_ref = ray.put(critic_weights)
        weight_td3, weight_hjb, weight_bdr = loss_weights
        #weight_delta = 1.0
        reward_window = deque(maxlen=100)
        q0_window     = deque(maxlen=100)
        
        ######################################
        # Initial exploration
        ######################################
        if initial_exploration_policy not in ("random", "policy"):
            raise ValueError(
                "initial_exploration_policy must be 'random' or 'policy', "
                f"got {initial_exploration_policy!r}."
            )
        use_random_initial = initial_exploration_policy == "random"
        initial_actor_ref = None if use_random_initial else actor_weights_ref
        initial_critic_ref = None if use_random_initial else critic_weights_ref
        worker_tasks = [
            worker.rollout.remote(
                initial_actor_ref,
                initial_critic_ref,
                random_action=use_random_initial,
            )
            for worker in workers
        ]
        buffer_size = 0
    
        while buffer_size < initial_exploration_num:
            finished, worker_tasks = ray.wait(worker_tasks, num_returns=1)
            worker_id, transitions, logs = ray.get(finished[0])
            buffer_size = ray.get( learner.add_memory.remote(transitions) )
            worker_tasks.append(
                workers[worker_id].rollout.remote(
                    initial_actor_ref,
                    initial_critic_ref,
                    random_action=use_random_initial,
                )
            )
        
        ##########################################
        # Start async learner process
        ##########################################
        learner_task = learner.learn_once.remote(
            minibatch_size=minibatch_size,
            loss_weights=loss_weights,
            num_collocations=num_collocations,
        )
        
        ##########################################
        # Training loop
        ##########################################    
        start, end = (agent.itr, agent.itr+num_iterations)
        pbar = tqdm(range(start+1, end+1), ascii=True, unit='updates') if verbose >= 1 else None
        update_count = start
        
        while update_count < end:
            ######################
            # Worker
            ######################
            finished_workers, worker_tasks = ray.wait(worker_tasks, num_returns=1, timeout=0.0)    
            if finished_workers:
                worker_id, transitions, logs = ray.get(finished_workers[0])        
                learner.add_memory.remote(transitions)
                reward_window.append(logs["episode_reward"])
                q0_window.append(logs["episode_q0"])
                worker_tasks.append( workers[worker_id].rollout.remote(actor_weights_ref, critic_weights_ref) )
        
            ###################
            # Learner
            ###################
            finished_learner, _ = ray.wait([learner_task], timeout=0.0)        
            if finished_learner:
                
                # Learner results
                loss = ray.get(finished_learner[0])
                actor_weights, critic_weights = ray.get(learner.get_network_weights.remote())
                actor_weights_ref = ray.put(actor_weights)
                critic_weights_ref = ray.put(critic_weights)
    
                update_count += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({
                        "td": f"{loss['td']:.2e}",
                        "hjb": f"{loss['hjb']:.2e}",
                    })
    
                # Log
                if update_count % log_freq == 0 and log_dir is not None:
                    
                    if loss["td"] == 0:
                        loss["td"] = ray.get(learner.evaluate_td_loss.remote(minibatch_size))

                    if loss["hjb"] == 0 or loss["bdr"] == 0:
                        loss["hjb"], loss["bdr"] = ray.get(learner.evaluate_hjb_bdr_loss.remote(num_collocations)                        )
                                    
                    summary_writer.add_scalar("RL/Average Reward", np.mean(reward_window), update_count)
                    summary_writer.add_scalar("RL/Episode Q0",     np.nanmean(q0_window),  update_count)
                    summary_writer.add_scalar("Loss/RL",  loss["td"],  update_count)
                    summary_writer.add_scalar("Loss/HJB", loss["hjb"], update_count)
                    summary_writer.add_scalar("Loss/BDR", loss["bdr"], update_count)
                    summary_writer.add_scalar("Weights/RL",  weight_td3, update_count)
                    summary_writer.add_scalar("Weights/HJB", weight_hjb, update_count)
                    summary_writer.add_scalar("Weights/BDR", weight_bdr, update_count)
                    summary_writer.flush()
    
                if update_count % checkpoint_freq == 0 or update_count == end:     
                    ray.get( learner.save.remote(summary_dir + f'/ckpt-{update_count}') )
                
                ###############################
                # Update Loss Weights
                ###############################
                if weight_schedule is not None:
                    if weight_schedule_time_base == "global":
                        schedule_count = update_count
                    elif weight_schedule_time_base == "local":
                        schedule_count = update_count - start
                    else:
                        raise ValueError(
                            "weight_schedule_time_base must be 'global' or 'local', "
                            f"got {weight_schedule_time_base!r}."
                        )
                    center    = weight_schedule["center"]
                    sharpness = weight_schedule["sharpness"]
                    w_start   = np.asarray(weight_schedule["initial"], dtype=float)
                    w_end     = np.asarray(weight_schedule["final"],   dtype=float)

                    sigm = 1.0 / (1.0 + np.exp(-sharpness * (schedule_count - center)))
                    if sigm < 0.01:
                        sigm = 0.0
                    elif sigm > 0.99:
                        sigm = 1.0         
                    weights = w_start + (w_end - w_start) * sigm
                    weight_td3, weight_hjb, weight_bdr = weights.tolist()
                
                #####################################
                # Restart learner process 
                #####################################
                learner_task = learner.learn_once.remote(
                    minibatch_size=minibatch_size,
                    loss_weights=(weight_td3, weight_hjb, weight_bdr),
                    num_collocations=num_collocations,
                )
            
        if pbar is not None:
            pbar.close()
    
    finally:
        print('Closing ray instance')
        ray.shutdown()     
        
        
        
        
