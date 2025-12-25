import gymnasium as gym
import numpy as np
from tqdm import tqdm

def train(train_episodes, env, agent):


    for episode in tqdm(range(train_episodes)):

        episode_over = False
        observation, info = env.reset()

        while not episode_over:
            

            action = agent.get_action(observation)

            next_observation, reward, terminated, truncated, info = env.step(action)

            agent.update(
                observation = observation,
                action = action,
                reward = reward,
                terminated = terminated,
                next_observation = next_observation
            )

            episode_over = terminated or truncated
            # discrete_observation = discrete_next_observation
            observation = next_observation

        agent.e_decay()

    return agent

def test(test_episodes, env, agent):

    total_rewards = []
    success = 0
    old_epsilon = agent.epsilon
    agent.epsilon = 0

    for episode in tqdm(range(test_episodes)):

        observation, info = env.reset()
        episode_reward = 0
        episode_over = False

        while not episode_over:

            action = agent.get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            # env.render()

            episode_reward += reward

            episode_over = terminated or truncated

        if terminated:
            success += 1
            
        total_rewards.append(episode_reward)

    agent.epsilon = old_epsilon

    average_reward = np.mean(total_rewards)

    print("==============================================")
    print(f"Test results over {test_episodes} episodes :")
    print(f"Successful Episodes : {success}")
    print(f"Unsuccessful Episodes : {test_episodes - success}")
    print(f"Average Reward : {average_reward:.3f}")
    print("==============================================")
