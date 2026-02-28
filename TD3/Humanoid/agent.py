import torch
from torch import nn
from collections import deque
import random
import numpy as np


class Actor(nn.Module):

    def __init__(self, observation_dims, action_dims, hidden_dims, max_action):
        super().__init__()
        self.layer1 = nn.Linear(observation_dims,hidden_dims)
        self.layer2 = nn.Linear(hidden_dims,hidden_dims)
        self.layer3 = nn.Linear(hidden_dims,action_dims)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.max_action = max_action

    def forward(self,x):

        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))

        return self.max_action*self.tanh(self.layer3(x))
    
class Critic(nn.Module):

    def __init__(self, observation_dims, action_dims, hidden_dims):
        super().__init__()
        self.layer1 = nn.Linear(observation_dims + action_dims,hidden_dims)
        self.layer2 = nn.Linear(hidden_dims,hidden_dims)
        self.layer3 = nn.Linear(hidden_dims,1)
        self.relu = nn.ReLU()

    def forward(self,observation,action):

        x = torch.cat([observation, action], dim=1)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))

        return self.layer3(x) #Deterministic q_value
    
class ReplayBuffer:

    def __init__(self,buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def push(self,observation,action,reward,next_observation,done):
        self.buffer.append((observation,action,reward,next_observation,done)) 

    def sample_experiences(self,batch_size):
        batch = random.sample(self.buffer,batch_size)

        observations,actions,rewards,next_observations,dones = map(np.array, zip(*batch))

        return (

            torch.FloatTensor(observations),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_observations),
            torch.FloatTensor(dones)

        )

    def __len__(self):
        return len(self.buffer)
    
    
class HumanoidTD3Agent:
        
    def __init__(
            self,
            env,
            actor_learning_rate,
            critic_learning_rate,
            hidden_dims,
            buffer_size,
            batch_size,
            tau,
            exploration_noise,
            actor_delay_freq,
            discount_factor = 0.95,
    ):
    
        self.env = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.observation_dims = env.observation_space.shape[0]
        self.action_dims = env.action_space.shape[0]
        self.max_action = env.action_space.high[0]

        self.buffer = ReplayBuffer(buffer_size)

        self.actor = Actor(self.observation_dims,self.action_dims,hidden_dims,self.max_action).to(self.device)
        self.actor_target = Actor(self.observation_dims,self.action_dims,hidden_dims,self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(),lr=actor_learning_rate)

        self.critic1 = Critic(self.observation_dims,self.action_dims,hidden_dims).to(self.device)
        self.critic1_target = Critic(self.observation_dims,self.action_dims,hidden_dims).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = torch.optim.Adam(params=self.critic1.parameters(),lr=critic_learning_rate)

        self.critic2 = Critic(self.observation_dims,self.action_dims,hidden_dims).to(self.device)
        self.critic2_target = Critic(self.observation_dims,self.action_dims,hidden_dims).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = torch.optim.Adam(params=self.critic2.parameters(),lr=critic_learning_rate)

        self.batch_size  = batch_size

        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.discount_factor = discount_factor
        self.tau = tau
        self.exploration_noise = exploration_noise
        self.actor_delay_freq = actor_delay_freq
        self.steps = 0
        
    def get_action(self, observation, add_noise=True):
            
        with torch.no_grad():
            observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action = self.actor(observation).cpu().data.numpy().flatten()

            if add_noise :

                noise = np.random.normal(0, self.exploration_noise, self.action_dims)
                action = action + noise

        return np.clip(action, -self.max_action, self.max_action)
        
    def update(self):

        if len(self.buffer) >= self.batch_size:
              
            observations,actions,rewards,next_observations,dones = self.buffer.sample_experiences(self.batch_size)

            observations = observations.to(self.device)
            actions = actions.view(-1, self.action_dims).to(self.device)
            rewards = rewards.unsqueeze(1).to(self.device)
            next_observations = next_observations.to(self.device)
            dones =  dones.unsqueeze(1).to(self.device)

            #Update Critic
            with torch.no_grad():

                next_actions = self.actor_target(next_observations)

                #Target Policy Smoothing
                noise = (torch.randn(size=next_actions.size(), device=next_actions.device)*0.2).clamp(-0.5, 0.5)
                next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)

                target_q_values1 = self.critic1_target(next_observations, next_actions)
                target_q_values2 = self.critic2_target(next_observations, next_actions)
                bellman = rewards + (self.discount_factor*(1-dones)*(torch.min(target_q_values1, target_q_values2))) #Twin Critic

            current_q_values1 = self.critic1(observations, actions)
            current_q_values2 = self.critic2(observations, actions)

            critic1_loss = nn.MSELoss()(current_q_values1,bellman)
            critic2_loss = nn.MSELoss()(current_q_values2,bellman)

            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            #Soft Target Update

            self.steps +=  1

            if self.steps % self.actor_delay_freq == 0 : #Delayed Actor Update

                #Update Actor

                predicted_actions = self.actor(observations)
                actor_loss = -(self.critic1(observations, predicted_actions)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

                for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

                for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def store_experience(self,observation,action,reward,next_observation,done):
        self.buffer.push(observation,action,reward,next_observation,done)