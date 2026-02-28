import gymnasium as gym
import torch
from agent import HalfCheetahSACAgent
from train_test import train,test
import numpy as np
import matplotlib.pyplot as plt

#Environment

train_env = gym.make("HalfCheetah-v5")
test_env = gym.make("HalfCheetah-v5")

#Parameters

train_episodes = 1000
test_episodes = 10
learning_rate = 0.001
tau = 0.005
temperature = 0.2
epsilon = 1e-6
discount_factor = 0.99
hidden_dims = 256
buffer_size = 1000000
batch_size = 256
warmup_steps = 10000

train_env = gym.wrappers.RecordEpisodeStatistics(train_env,buffer_length=train_episodes)

#Half-Cheetah SAC Agent

agent = HalfCheetahSACAgent(
    env=train_env,
    learning_rate=learning_rate,
    hidden_dims=hidden_dims,
    buffer_size=buffer_size,
    batch_size=batch_size,
    tau=tau,
    temperature=temperature,
    epsilon=epsilon,
    discount_factor=discount_factor
)

#Training
print("Training...")
trained_agent = train(train_episodes=train_episodes,env=train_env,agent=agent,warmup_steps=warmup_steps)
train_env.close()

#Testing
print("Testing...")
test(test_episodes=test_episodes,env=test_env,agent=trained_agent)
test_env.close()

#Saving model
torch.save(agent.actor.state_dict(), "halfcheetah_sac_actor.pth")
torch.save(agent.critic1.state_dict(), "halfcheetah_sac_critic1.pth")
torch.save(agent.critic2.state_dict(), "halfcheetah_sac_critic2.pth")
print("Half-Cheetah SAC Models saved successfully.")
