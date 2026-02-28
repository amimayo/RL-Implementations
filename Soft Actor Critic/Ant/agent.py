import torch
from torch import nn
from collections import deque
import random
import numpy as np


class Actor(nn.Module):

    def __init__(self, observation_dims, action_dims, hidden_dims):
        super().__init__()
        self.layer1 = nn.Linear(observation_dims,hidden_dims)
        self.layer2 = nn.Linear(hidden_dims,hidden_dims)
        self.relu = nn.ReLU()

        self.fc_mean = nn.Linear(hidden_dims, action_dims)
        self.fc_logvar = nn.Linear(hidden_dims, action_dims)

    def forward(self,x, return_log_prob=False, epsilon=1e-6):

        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))

        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        if not return_log_prob :

            return mean, logvar

        std = torch.exp(0.5*logvar)
        dist = torch.distributions.Normal(mean, std)
        e = torch.randn(size=std.size(), device=std.device)

        action = torch.tanh(mean + (e*std))

        log_prob = dist.log_prob(mean + (e*std)) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob



    
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

        return self.layer3(x)
    
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
    
    
class AntSACAgent:
        
    def __init__(
            self,
            env,
            learning_rate,
            hidden_dims,
            buffer_size,
            batch_size,
            tau,
            temperature,
            epsilon,
            discount_factor = 0.95,
    ):
    
        self.env = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.observation_dims = env.observation_space.shape[0]
        self.action_dims = env.action_space.shape[0]

        self.buffer = ReplayBuffer(buffer_size)

        self.actor = Actor(self.observation_dims,self.action_dims,hidden_dims).to(self.device)
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(),lr=learning_rate)

        self.critic1 = Critic(self.observation_dims,self.action_dims,hidden_dims).to(self.device)
        self.critic1_target = Critic(self.observation_dims,self.action_dims,hidden_dims).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = torch.optim.Adam(params=self.critic1.parameters(),lr=learning_rate)

        self.critic2 = Critic(self.observation_dims,self.action_dims,hidden_dims).to(self.device)
        self.critic2_target = Critic(self.observation_dims,self.action_dims,hidden_dims).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = torch.optim.Adam(params=self.critic2.parameters(),lr=learning_rate)

        self.batch_size  = batch_size

        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.tau = tau
        self.temperature = temperature
        self.epsilon = epsilon

    def reparameterize(self, mean, logvar):

        std = torch.exp(0.5*logvar)
        e = torch.randn(size=std.size(), device=std.device)

        action = torch.tanh(mean + (e*std))

        return action
        
    def get_action(self, observation, test=False):
            
        with torch.no_grad():
            observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            mean, logvar = self.actor(observation)

            if test :

                action = torch.tanh(mean)

            else :

                action  = self.reparameterize(mean, logvar)

        return action.cpu().numpy().flatten()
        
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

                next_actions, next_log_prob = self.actor.forward(next_observations, return_log_prob=True, epsilon=self.epsilon)
                target_q_values1 = self.critic1_target(next_observations, next_actions)
                target_q_values2 = self.critic2_target(next_observations, next_actions)
                bellman = rewards + self.discount_factor*(1-dones)*((torch.min(target_q_values1, target_q_values2)) - (self.temperature*next_log_prob))

            current_q_values1 = self.critic1(observations, actions)
            current_q_values2 = self.critic2(observations, actions)
            critic1_loss = nn.MSELoss()(current_q_values1,bellman.detach())
            critic2_loss = nn.MSELoss()(current_q_values2,bellman.detach())

            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            #Update Actor

            predicted_actions, log_prob = self.actor.forward(observations, return_log_prob=True, epsilon=self.epsilon)

            predicted_q_values1 = self.critic1(observations, predicted_actions)
            predicted_q_values2 = self.critic2(observations, predicted_actions)

            actor_loss = -((torch.min(predicted_q_values1, predicted_q_values2)) - (self.temperature*log_prob)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            #Soft Target Update

            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def store_experience(self,observation,action,reward,next_observation,done):
        self.buffer.push(observation,action,reward,next_observation,done)