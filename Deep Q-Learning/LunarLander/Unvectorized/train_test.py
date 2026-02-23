import gymnasium as gym
import numpy as np
from tqdm import tqdm

def train(train_episodes, env, agent,warmup_steps):

    observation,info = env.reset()

    print("Warmup...") #Warmup
    while len(agent.buffer) < warmup_steps:
            action = env.action_space.sample()
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_experience(observation, action, reward, next_observation, done)
            observation = next_observation if not done else env.reset()[0]

    print("Warmup Finished.")
    
    print("Training")

    for episode in tqdm(range(train_episodes)):

        episode_over = False
        observation, info = env.reset()

        while not episode_over:

            action = agent.get_action(observation)

            next_observation, reward, terminated, truncated, info = env.step(action)

            episode_over = terminated or truncated

            agent.store_experience(observation,action,reward,next_observation,episode_over)
            
            observation = next_observation

            agent.update()


        agent.e_decay()

    return agent

def test(test_episodes, env, agent):

    total_rewards = []
    old_epsilon = agent.epsilon
    agent.epsilon = 0
    successes = 0

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
            
        total_rewards.append(episode_reward)
        if episode_reward > 200:
            successes += 1

    agent.epsilon = old_epsilon

    success_rate = successes/test_episodes
    average_reward = np.mean(total_rewards)

    print("==============================================")
    print(f"Test results over {test_episodes} episodes :")
    print(f"Success Rate : {success_rate:.1%}")
    print(f"Fail Rate : {(1-success_rate):.1%}")
    print(f"Successful Landings : {successes}")
    print(f"Unsuccessful Episodes : {test_episodes - successes}")
    print(f"Average Reward : {average_reward:.3f}")
    print("==============================================")
