import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import torch
from agent import PongDDQNAgent
from train_test import train,test
import numpy as np
import matplotlib.pyplot as plt

def main():

    #Environment

    num_envs = 8

    def make_env(id):
            env = gym.make(id, frameskip=1)
            env = AtariPreprocessing(env=env, frame_skip=4, screen_size=84, grayscale_obs=True, scale_obs=False) #Convert to (84, 84) Grayscale
            env = FrameStackObservation(env=env, stack_size=4) #Stack 4 Frames

            return env 

    train_env = gym.vector.AsyncVectorEnv([lambda : make_env("ALE/Pong-v5") for _ in range(num_envs)])
    test_env = make_env("ALE/Pong-v5")

    #Parameters

    train_steps = 300_000
    train_episodes = 1000
    test_episodes = 10
    learning_rate = 0.0001
    initial_epsilon = 1
    final_epsilon = 0.01
    epsilon_decay = (initial_epsilon - final_epsilon)/(train_steps/2)
    discount_factor = 0.99
    hidden_dims = 512
    buffer_size = 100000
    batch_size = 64
    update_target_freq = 5000
    warmup_steps = 10000

    # Pong Double Deep-Q-Learning Agent

    agent = PongDDQNAgent(
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
    trained_agent = train(train_steps=train_steps,warmup_steps=warmup_steps,env=train_env,agent=agent)
    train_env.close()

    #Testing
    print("Testing...")
    test(test_episodes=test_episodes,env=test_env,agent=trained_agent)
    test_env.close()

    #Saving model
    torch.save(agent.qpolicy_network.state_dict(), "pong_vec_ddqn_model.pth")
    print("Pong DDQN Model saved successfully.")

if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    main()
