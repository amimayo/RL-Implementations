import gymnasium as gym
from agent import TaxiAgent
from train_test import train,test
import numpy as np
import matplotlib.pyplot as plt

#Environment

train_env = gym.make("Taxi-v3")
test_env = gym.make("Taxi-v3")

#Parameters

train_episodes = 10_000
test_episodes = 100
learning_rate = 0.1
initial_epsilon = 1
epsilon_decay = (initial_epsilon)/(train_episodes/2)
final_epsilon = 0.005
discount_factor = 0.99
train_env = gym.wrappers.RecordEpisodeStatistics(train_env, buffer_length=train_episodes)

#FrozenLake Q-Learning Agent

agent = TaxiAgent(
    env=train_env,
    learning_rate=learning_rate,
    initial_epsilon=initial_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor=discount_factor
)

#Training
print("Training...")
trained_agent = train(train_episodes=train_episodes,env=train_env,agent=agent)
train_env.close()
# for state in range(train_env.observation_space.n):
#     print(f"State {state}: {agent.q_values[state]}")


#Testing
print("Testing...")
test(test_episodes=test_episodes,env=test_env,agent=trained_agent)
test_env.close()
