import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, obs_dims, action_dims, hidden_dims):
        super().__init__()
        self.layer1 = nn.Linear(obs_dims, hidden_dims)
        self.layer2 = nn.Linear(hidden_dims, hidden_dims)
        self.mean_layer = nn.Linear(hidden_dims, action_dims)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        self.action_std = nn.Parameter(torch.zeros(action_dims))

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        action_mean = self.tanh(self.mean_layer(x))
        
        action_std = torch.exp(self.action_std) 

        return action_mean, action_std #Return mean and std for continuous spaces
    

class Critic(nn.Module):
    def __init__(self, obs_dims, hidden_dims):
        super().__init__()
        self.layer1 = nn.Linear(obs_dims, hidden_dims)
        self.layer2 = nn.Linear(hidden_dims, hidden_dims)
        self.layer3 = nn.Linear(hidden_dims, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.layer3(x)
    

class Buffer:
    def __init__(self):
        self.observations, self.actions, self.observation_values = [], [], []
        self.action_logprobs, self.rewards, self.dones = [], [], []

    def clear(self):
        self.__init__()

class PPOAgent:
    def __init__(self, env, learning_rate, gae_lambda, discount_factor, hidden_dims, batch_size, clip, device):
        self.device = device
        self.batch_size = batch_size
        self.clip = clip
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        
        obs_dims = env.observation_space.shape[0]
        action_dims = env.action_space.shape[0] 
        
        self.buffer = Buffer()
        
        self.actor = Actor(obs_dims, action_dims, hidden_dims).to(device)
        self.critic = Critic(obs_dims, hidden_dims).to(device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

    def get_action(self, observation):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            mean, std = self.actor(obs_tensor)
            value = self.critic(obs_tensor)

            dist = torch.distributions.Normal(mean, std) #Gaussian/normal distribution 
            action = dist.sample() 
            
            action_logprob = dist.log_prob(action).sum(dim=-1)

            self.buffer.observations.append(obs_tensor.cpu().squeeze())
            self.buffer.actions.append(action.cpu().squeeze())
            self.buffer.observation_values.append(value.cpu().squeeze())
            self.buffer.action_logprobs.append(action_logprob.cpu().squeeze())
            
            return action.cpu().numpy().flatten()

    def compute_gae(self, next_value):

        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float32).to(self.device)
        values = torch.tensor(self.buffer.observation_values, dtype=torch.float32).to(self.device)
        
        next_values = torch.cat([values[1:], next_value.view(-1)])
        
        advantages = torch.zeros_like(rewards).to(self.device)
        gae = 0
        
        # GAE math
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.discount_factor * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.discount_factor * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            
        returns = advantages + values
        return advantages, returns
    
    def update(self, epochs):
        
        old_states = torch.stack(self.buffer.observations).to(self.device)
        old_actions = torch.stack(self.buffer.actions).to(self.device)
        old_logprobs = torch.stack(self.buffer.action_logprobs).to(self.device)

        with torch.no_grad():
            next_value = self.critic(old_states[-1].unsqueeze(0))
            advantages, returns = self.compute_gae(next_value)
            #Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(epochs):
            indices = np.random.permutation(len(old_states))
            
            for start in range(0, len(old_states), self.batch_size):
                idx = indices[start:start + self.batch_size]    

                mean, std = self.actor(old_states[idx])
                dist = torch.distributions.Normal(mean, std)
                
                new_logprobs = dist.log_prob(old_actions[idx]).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                #Clipped Objective Function
                ratios = torch.exp(new_logprobs - old_logprobs[idx])
                surr1 = ratios * advantages[idx]
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages[idx]

                actor_loss = -torch.min(surr1, surr2).mean() - (0.01 * entropy)
                critic_loss = F.mse_loss(self.critic(old_states[idx]).squeeze(), returns[idx])

                #Update Actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                #Update Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
       
        self.buffer.clear()