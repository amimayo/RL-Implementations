import gymnasium as gym
import torch
from agent import LunarLanderDQNAgent
from train_test import train,test
import numpy as np
import matplotlib.pyplot as plt

def main():

    #Environment

    num_envs = 8
    train_env = gym.make_vec("LunarLander-v3",num_envs=num_envs,vectorization_mode="async")
    test_env = gym.make("LunarLander-v3")

    #Parameters

    train_steps = 500_000
    train_episodes = 1000
    test_episodes = 10
    learning_rate = 0.001
    initial_epsilon = 1
    final_epsilon = 0.01
    epsilon_decay = (initial_epsilon - final_epsilon)/(train_steps/2)
    discount_factor = 0.9999
    hidden_dims = 512
    buffer_size = 100000
    batch_size = 64
    update_target_freq = 5000
    warmup_steps = 10000

    # train_env = gym.wrappers.RecordEpisodeStatistics(train_env,buffer_length=train_episodes)

    #LunarLander Deep-Q-Learning Agent

    agent = LunarLanderDQNAgent(
        env=train_env,
        learning_rate=learning_rate,
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        hidden_dims=hidden_dims,
        buffer_size=buffer_size,
        batch_size=batch_size,
        update_target_freq = update_target_freq,
        num_envs = num_envs,
        discount_factor=discount_factor
    )

    #Training
    print("Training...")
    trained_agent = train(train_steps=train_steps,env=train_env,agent=agent)
    train_env.close()

    #Testing
    print("Testing...")
    test(test_episodes=test_episodes,env=test_env,agent=trained_agent)
    test_env.close()

    #Saving model
    torch.save(agent.qpolicy_network.state_dict(), "lunarlander_vec_dqn_model.pth")
    print("LunarLander DQN Model saved successfully.")

if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    main()
