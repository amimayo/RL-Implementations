import gymnasium as gym
import numpy as np
from tqdm import tqdm

def train(train_steps, env, agent):

    num_envs = env.num_envs
    observations, infos = env.reset()

    for step in tqdm(range(train_steps)):

            actions = agent.get_actions(observations)

            next_observations, rewards, terminateds, truncateds, infos = env.step(actions)

            episode_overs = np.logical_or(terminateds, truncateds)

            agent.store_experience(observations,actions,rewards,next_observations,episode_overs)
            
            agent.update()

            agent.e_decay()

            observations = next_observations

    return agent

def test(test_episodes, env, agent):

    total_rewards = []
    old_epsilon = agent.epsilon
    agent.epsilon = 0

    for episode in tqdm(range(test_episodes)):

        observation, info = env.reset()
        episode_reward = 0
        successes = 0
        episode_over = False

        while not episode_over:

            action = agent.get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            # env.render()

            if reward > 200:
                successes += 1

            episode_reward += reward

            episode_over = terminated or truncated
            
        total_rewards.append(episode_reward)

    agent.epsilon = old_epsilon

    success_rate = np.mean(np.array(total_rewards) > 200)
    average_reward = np.mean(total_rewards)

    print("==============================================")
    print(f"Test results over {test_episodes} episodes :")
    print(f"Success Rate : {success_rate:.1%}")
    print(f"Fail Rate : {(1-success_rate):.1%}")
    print(f"Successful Landings : {successes}")
    print(f"Unsuccessful Episodes : {test_episodes - successes}")
    print(f"Average Reward : {average_reward:.3f}")
    print("==============================================")
