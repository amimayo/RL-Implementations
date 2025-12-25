import gymnasium as gym
import numpy as np
from tqdm import tqdm


def train(train_episodes, agent, env):

    for episode in tqdm(range(train_episodes)):

        episode_done = False
        observation, info = env.reset()

        while not episode_done:

            action = agent.get_action(observation)

            next_observation, reward, terminated, truncated, info = env.step(action)

            agent.update(
                observation=observation,
                action=action,
                reward=reward,
                terminated=terminated,
                next_observation=next_observation
            )

            episode_done = terminated or truncated

            observation = next_observation
        
        agent.e_decay()

    return agent

def test(test_episodes, agent, env):

        total_rewards = []

        old_epsilon = agent.epsilon
        agent.epsilon = 0.0

        for episode in tqdm(range(test_episodes)):

            observation, info = env.reset()
            episode_reward = 0
            episode_done = False

            while not episode_done:

                action = agent.get_action(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                env.render()
                episode_reward += reward
                episode_done = terminated or truncated

            total_rewards.append(episode_reward)

        agent.epsilon = old_epsilon

        win_rate = np.mean(np.array(total_rewards) > 0)
        loss_rate = np.mean(np.array(total_rewards) < 0)
        draw_rate = np.mean(np.array(total_rewards) == 0)
        average_reward = np.mean(total_rewards)

        print("==============================================")
        print(f"Test results over {test_episodes} episodes :")
        print(f"Win Rate : {win_rate:.1%}")
        print(f"Loss Rate : {loss_rate:.1%}")
        print(f"Draw Rate : {draw_rate:.1%}")
        print(f"Average Reward : {average_reward:.3f}")
        print("==============================================")


            




