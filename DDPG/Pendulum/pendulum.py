import gymnasium as gym
import torch
from agent import PendulumDDPGAgent
from train_test import train,test
import numpy as np
import matplotlib.pyplot as plt

#Environment

train_env = gym.make("Pendulum-v1")
test_env = gym.make("Pendulum-v1",render_mode="human")

#Parameters

train_episodes = 300
test_episodes = 10
actor_learning_rate = 0.0001
critic_learning_rate = 0.001
tau = 0.005
exploration_noise = 0.2
discount_factor = 0.99
hidden_dims = 128
buffer_size = 100000
batch_size = 64
warmup_steps = 10000

train_env = gym.wrappers.RecordEpisodeStatistics(train_env,buffer_length=train_episodes)

#Pendulum DDPG Agent

agent = PendulumDDPGAgent(
    env=train_env,
    actor_learning_rate=actor_learning_rate,
    critic_learning_rate=critic_learning_rate,
    hidden_dims=hidden_dims,
    buffer_size=buffer_size,
    batch_size=batch_size,
    tau=tau,
    exploration_noise=exploration_noise,
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
torch.save(agent.actor.state_dict(), "pendulum_ddpg_actor.pth")
torch.save(agent.critic.state_dict(), "pendulum_ddpg_critic.pth")
print("Pendulum DDPG Models saved successfully.")

