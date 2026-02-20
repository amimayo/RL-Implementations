import gymnasium as gym
import numpy as np
from tqdm import tqdm

def train(train_episodes, env, agent):


    for episode in tqdm(range(train_episodes)):

        episode_over = False
        observation, info = env.reset()
        action = agent.get_action(observation)

        while not episode_over:

            next_observation, reward, terminated, truncated, info = env.step(action)

            next_action = agent.get_action(next_observation)


            agent.update(
                observation = observation,
                action = action,
                reward = reward,
                terminated = terminated,
                next_observation = next_observation,
                next_action = next_action
            )

            episode_over = terminated or truncated
            observation = next_observation
            action = next_action

        agent.e_decay()

    return agent

def test(test_episodes, env, agent):

    total_rewards = []
    success_rides = 0
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
            
            if episode_reward == -17:
                success_rides += 1

            episode_over = terminated or truncated
            
        total_rewards.append(episode_reward)

    agent.epsilon = old_epsilon

    average_reward = np.mean(total_rewards)

    print("==============================================")
    print(f"Test results over {test_episodes} episodes :")
    print(f"Successful Rides : {success_rides}")
    print(f"Unsuccessful Rides : {test_episodes - success_rides}")
    print(f"Average Reward : {average_reward:.3f}")
    print("==============================================")






