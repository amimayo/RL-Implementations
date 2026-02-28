import gymnasium as gym
import torch
from agent import BipedalWalkerTD3Agent
from train_test import train,test
import numpy as np
import matplotlib.pyplot as plt

#Environment

train_env = gym.make("BipedalWalker-v3")
test_env = gym.make("BipedalWalker-v3")

#Parameters

train_episodes = 2000
test_episodes = 10
actor_learning_rate = 0.0001
critic_learning_rate = 0.001
tau = 0.005
exploration_noise = 0.1
discount_factor = 0.99
hidden_dims = 256
buffer_size = 1000000
batch_size = 256
actor_delay_freq = 2
warmup_steps = 10000

train_env = gym.wrappers.RecordEpisodeStatistics(train_env,buffer_length=train_episodes)

#Bipedal-Walker TD3 Agent

agent = BipedalWalkerTD3Agent(
    env=train_env,
    actor_learning_rate=actor_learning_rate,
    critic_learning_rate=critic_learning_rate,
    hidden_dims=hidden_dims,
    buffer_size=buffer_size,
    batch_size=batch_size,
    tau=tau,
    exploration_noise=exploration_noise,
    actor_delay_freq=actor_delay_freq,
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

 #Saving models
torch.save(agent.actor.state_dict(), "bipedalwalker_td3_actor.pth")
torch.save(agent.critic1.state_dict(), "bipedalwalker_td3_critic1.pth")
torch.save(agent.critic2.state_dict(), "bipedalwalker_td3_critic2.pth")
print("Bipedal-Walker TD3 Models saved successfully.")

