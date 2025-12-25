import gymnasium as gym
from agent import BlackJackAgent
from train_test import train,test
import numpy as np
import matplotlib.pyplot as plt

#Environment
train_env = gym.make("Blackjack-v1", sab=False)
test_env = gym.make("Blackjack-v1",natural=True, sab=True,render_mode="human")

#Parameters
train_episodes = 10_000
test_episodes = 1000
learning_rate = 0.01
initial_epsilon = 1.0
epsilon_decay = initial_epsilon/(train_episodes/2)
final_epsilon = 0.005
discount_factor = 0.99
train_env = gym.wrappers.RecordEpisodeStatistics(train_env, buffer_length=train_episodes)

#BlackJack Q-Learning Agent
agent = BlackJackAgent(
    learning_rate = learning_rate,
    initial_epsilon = initial_epsilon,
    epsilon_decay = epsilon_decay,
    final_epsilon = final_epsilon,
    discount_factor = discount_factor,
    env = train_env
)

#Training
print("Training...")
trained_agent = train(train_episodes=train_episodes,  agent=agent, env=train_env)
train_env.close()

#Testing
print("Testing...")
test(test_episodes=test_episodes, agent=trained_agent, env=test_env)
test_env.close()