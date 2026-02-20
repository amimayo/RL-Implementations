import gymnasium as gym
import torch
from agent import PPOAgent
from train_test import train, test

# Environment
train_env = gym.make("BipedalWalker-v3")
test_env = gym.make("BipedalWalker-v3", render_mode="human")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
train_episodes = 2500
test_episodes = 10
update_train_steps = 2048
learning_rate = 0.0003
gae_lambda = 0.96
hidden_dims = 256
batch_size = 64
clip = 0.2
discount_factor = 0.99
epochs = 10

train_env = gym.wrappers.RecordEpisodeStatistics(train_env, buffer_length=train_episodes)

# Bipedal-Walker PPO Agent
agent = PPOAgent(
   env=train_env,
   learning_rate=learning_rate,
   gae_lambda=gae_lambda,
   discount_factor=discount_factor,
   hidden_dims=hidden_dims,
   batch_size=batch_size,
   clip=clip,
   device=device
)

# Training
print(f"Training...")
trained_agent = train(train_episodes=train_episodes, update_train_steps=update_train_steps, epochs=epochs, env=train_env, agent=agent)
train_env.close()

# Testing
print("Testing...")
test(test_episodes=test_episodes, env=test_env, agent=trained_agent)
test_env.close()

#Saving models
torch.save(agent.actor.state_dict(), "bipedalwalker_ppo_actor.pth")
torch.save(agent.critic.state_dict(), "bipedalwalker_ppo_critic.pth")
print("Bipedal-Walker PPO Models saved successfully.")