import gymnasium as gym
import numpy as np
from tqdm import tqdm

def train(train_episodes, env, agent):


    for episode in tqdm(range(train_episodes)):

        episode_over = False
        observation, info = env.reset()

        while not episode_over:
            
            discrete_observation = agent.discretize(observation)

            action_idx = agent.get_action(discrete_observation)

            action = np.array([agent.discretized_actions[action_idx]])

            next_observation, reward, terminated, truncated, info = env.step(action)

            discrete_next_observation = agent.discretize(next_observation)

            agent.update(
                observation = discrete_observation,
                action_idx = action_idx,
                reward = reward,
                terminated = terminated,
                next_observation = discrete_next_observation
            )

            episode_over = terminated or truncated
           
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

            discrete_observation = agent.discretize(observation)
            action_idx = agent.get_action(discrete_observation)
            action = np.array([agent.discretized_actions[action_idx]])
            observation, reward, terminated, truncated, info = env.step(action)
            # env.render()

            episode_reward += reward

            episode_over = terminated or truncated

        if episode_reward >= -200:
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
