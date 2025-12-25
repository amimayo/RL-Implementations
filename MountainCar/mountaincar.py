import gymnasium as gym
from agent import MountainCarAgent
from train_test import train,test
import numpy as np
import matplotlib.pyplot as plt

#Environment

train_env = gym.make("MountainCar-v0")
test_env = gym.make("MountainCar-v0")

#Parameters

train_episodes = 1000
test_episodes = 10
learning_rate = 0.1
initial_epsilon = 1
epsilon_decay = (initial_epsilon)/(train_episodes/2)
final_epsilon = 0.005
discount_factor = 0.99
n_bins = (10,10)
n_tilings = 8
train_env = gym.wrappers.RecordEpisodeStatistics(train_env, buffer_length=train_episodes)

#FrozenLake Q-Learning Agent

agent = MountainCarAgent(
    env=train_env,
    learning_rate=learning_rate,
    initial_epsilon=initial_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor=discount_factor,
    n_bins=n_bins,
    n_tilings = n_tilings
)
#Training
print("Training...")
trained_agent = train(train_episodes=train_episodes,env=train_env,agent=agent)
train_env.close()

#Testing
print("Testing...")
test(test_episodes=test_episodes,env=test_env,agent=trained_agent)
test_env.close()
