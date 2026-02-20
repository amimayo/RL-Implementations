import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):
    def __init__(self, observation_dims, action_dims, hidden_dims):
        super().__init__()
        self.layer1 = nn.Linear(observation_dims, hidden_dims)
        self.layer2 = nn.Linear(hidden_dims, hidden_dims)
        self.layer3 = nn.Linear(hidden_dims, action_dims)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        # Returns probabilities, not log-probs
        return F.softmax(self.layer3(x), dim=-1)
    

class Critic(nn.Module):
    def __init__(self, observation_dims, hidden_dims):
        super().__init__()
        self.layer1 = nn.Linear(observation_dims, hidden_dims)
        self.layer2 = nn.Linear(hidden_dims, hidden_dims)
        self.layer3 = nn.Linear(hidden_dims, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.layer3(x)
    

class Buffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.observation_values = []
        self.action_logprobs = []
        self.rewards = []
        self.dones = []

    def clear(self):
        del self.observations[:]
        del self.actions[:]
        del self.observation_values[:]
        del self.action_logprobs[:]
        del self.rewards[:]
        del self.dones[:]


class PPOAgent:
    def __init__(
            self, env, learning_rate, gae_lambda, discount_factor, 
            hidden_dims, batch_size, clip, device
    ):
        self.env = env
        self.device = device
        self.observation_dims = env.observation_space.shape[0]
        self.action_dims = env.action_space.n
        self.batch_size = batch_size
        self.clip = clip
        
        self.buffer = Buffer()
        
        self.actor = Actor(self.observation_dims, self.action_dims, hidden_dims).to(self.device)
        self.critic = Critic(self.observation_dims, hidden_dims).to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=learning_rate)
        
        self.lr = learning_rate
        self.gae_lambda = gae_lambda
        self.discount_factor = discount_factor

    def get_action(self, observation):
        with torch.no_grad():
            observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            # Actor returns probabilities
            action_probs = self.actor(observation)
            observation_value = self.critic(observation)

            action_dist = torch.distributions.Categorical(probs=action_probs)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)

            self.buffer.observations.append(observation.detach().cpu())
            self.buffer.actions.append(action.detach().cpu())
            self.buffer.observation_values.append(observation_value.detach().cpu())
            self.buffer.action_logprobs.append(action_log_prob.detach().cpu())
            
            return action.item()

    def compute_gae(self, next_observation_value):
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).unsqueeze(0).to(self.device)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float32).unsqueeze(0).to(self.device)

        observation_values = torch.cat(self.buffer.observation_values).view(1,-1).to(self.device)
        new_observation_values = torch.cat([observation_values, next_observation_value.view(1,1)], dim=1)

        batch_size = rewards.shape[0]
        t_steps = rewards.shape[1]
        
        gae = torch.zeros(batch_size, ).to(self.device)
        advantages = torch.zeros(batch_size, t_steps).to(self.device)

        for t in reversed(range(t_steps)):
            
            temporal_diff = rewards[:,t] + (self.discount_factor * new_observation_values[:,t+1] * (1-dones[:,t])) - new_observation_values[:,t]
           
            gae = temporal_diff + (self.discount_factor * self.gae_lambda * gae * (1-dones[:,t]))
            advantages[:,t] = gae
         
        returns = advantages + observation_values
        return advantages.squeeze(), returns.squeeze()
    
    def update(self, epochs):
        old_observations = torch.cat(self.buffer.observations).to(self.device)
        old_actions = torch.cat(self.buffer.actions).to(self.device)
        old_action_logprobs = torch.cat(self.buffer.action_logprobs).to(self.device)

        with torch.no_grad():
            last_observation = old_observations[-1].unsqueeze(0)
            next_observation_value = self.critic(last_observation)
            advantages, returns = self.compute_gae(next_observation_value)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = old_observations.size(0)

        for epoch in range(epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)    

            for start in range(0, dataset_size, self.batch_size):
                idx = indices[start:start + self.batch_size]    

                probs = self.actor(old_observations[idx])
                dist = torch.distributions.Categorical(probs=probs)
                new_action_logprobs = dist.log_prob(old_actions[idx])
                entropy = dist.entropy().mean()

                # Clipped Objective Function
                ratios = torch.exp(new_action_logprobs - old_action_logprobs[idx])
                surr1 = ratios * advantages[idx]
                surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages[idx]

                actor_loss = -torch.min(surr1, surr2).mean() - (0.01 * entropy)

                new_observation_values = self.critic(old_observations[idx]).squeeze()
                critic_loss = F.mse_loss(new_observation_values, returns[idx])

                # Update Actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

       
        self.buffer.clear()