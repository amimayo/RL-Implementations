import torch
from torch import nn
from collections import deque
import random
import numpy as np


# class DQN(nn.Module):

#     def __init__(self, observation_dims, action_dims, hidden_dims):
#         super().__init__()
#         self.layer1 = nn.Linear(observation_dims,hidden_dims)
#         self.layer2 = nn.Linear(hidden_dims,hidden_dims)
#         self.layer3 = nn.Linear(hidden_dims,action_dims)
#         self.relu = nn.ReLU()

#     def forward(self,x):
#         x = self.relu(self.layer1(x))
#         x = self.relu(self.layer2(x))
#         return self.layer3(x)

#Todo

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 
    

class ReplayBuffer:

    def __init__(self,buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def push(self,observations,actions,rewards,next_observations,dones):
        
        for observation,action,reward,next_observation,done in zip(observations,actions,rewards,next_observations,dones):
            self.buffer.append((observation,action,reward,next_observation,done)) 

    def sample_experiences(self,batch_size):
        batch = random.sample(self.buffer,batch_size)

        observations,actions,rewards,next_observations,dones = map(np.array, zip(*batch))

        return (

            torch.FloatTensor(observations),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_observations),
            torch.FloatTensor(dones)

        )

    def __len__(self):
        return len(self.buffer)
    
    
class PongDDQNAgent:
        
    def __init__(
            self,
            env,
            learning_rate,
            initial_epsilon,
            epsilon_decay,
            final_epsilon,
            hidden_dims,
            buffer_size,
            batch_size,
            update_target_freq,
            num_envs,
            discount_factor = 0.95
    ):
    
        self.env = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_envs = num_envs
        self.observation_dims = env.single_observation_space.shape[0]
        self.action_dims = env.single_action_space.n

        self.buffer = ReplayBuffer(buffer_size)
        self.qpolicy_network = CNN(self.observation_dims,self.action_dims,hidden_dims).to(self.device)
        self.target_network = CNN(self.observation_dims,self.action_dims,hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.qpolicy_network.state_dict())
        self.optimizer = torch.optim.Adam(params=self.qpolicy_network.parameters(),lr=learning_rate)
        self.target_network.eval()

        self.steps = 0
        self.batch_size  = batch_size
        self.update_target_freq = update_target_freq

        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor

    def get_actions(self, observations):
        if np.random.random() < self.epsilon:
            return np.array([self.env.single_action_space.sample() for _ in range(self.num_envs)])
        else:
            with torch.no_grad():
                observations = torch.FloatTensor(observations).to(self.device)
                q_values = self.qpolicy_network(observations)
            return torch.argmax(q_values,dim=1).cpu().numpy()
        
    def get_action(self, observation):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                q_values = self.qpolicy_network(observation)
            return torch.argmax(q_values,dim=1).item()
        
    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        
        observations,actions,rewards,next_observations,dones = self.buffer.sample_experiences(self.batch_size)

        observations = observations.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        next_observations = next_observations.to(self.device)
        dones =  dones.unsqueeze(1).to(self.device)

        with torch.no_grad():

            #DDQN

            best_actions = self.qpolicy_network(next_observations).argmax(dim=1, keepdim=True)
            future_q_values = self.target_network(next_observations).gather(1, best_actions)
            target_q_values = rewards + (self.discount_factor*future_q_values*(1-dones))

        q_values = self.qpolicy_network(observations).gather(1, actions)
        loss = nn.MSELoss()(q_values,target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qpolicy_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps += self.num_envs
        if self.steps % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.qpolicy_network.state_dict())

    def store_experience(self,observations,actions,rewards,next_observations,dones):
        self.buffer.push(observations,actions,rewards,next_observations,dones)
        
    def e_decay(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)