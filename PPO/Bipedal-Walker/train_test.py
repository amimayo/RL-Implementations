import numpy as np
from tqdm import tqdm

def train(train_episodes, update_train_steps, epochs, env, agent):
    steps = 0 

    for episode in tqdm(range(train_episodes)):
        episode_done = False
        observation, info = env.reset()

        while not episode_done:
            steps += 1
            action = agent.get_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)

            agent.buffer.rewards.append(reward)
            agent.buffer.dones.append(terminated or truncated)

            if steps % update_train_steps == 0:
                agent.update(epochs)

            episode_done = terminated or truncated
            observation = next_observation

    return agent

def test(test_episodes, env, agent):
    total_rewards = []

    for episode in tqdm(range(test_episodes)):
        observation, info = env.reset()
        episode_reward = 0
        episode_done = False

        while not episode_done:
            action = agent.get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_done = terminated or truncated

        total_rewards.append(episode_reward)

    win_rate = np.mean(np.array(total_rewards) >= 300) 
    average_reward = np.mean(total_rewards)

    print("==============================================")
    print(f"Test results over {test_episodes} episodes :")
    print(f"Win Rate (Max Score) : {win_rate:.1%}")
    print(f"Average Reward : {average_reward:.3f}")
    print("==============================================")