import gymnasium as gym
import numpy as np
from tilecoding import TileCoder,TiledQTable 
from collections import defaultdict

class MountainCarAgent:

    def __init__(
            self,
            env : gym.Env,
            learning_rate : float,
            initial_epsilon : float,
            epsilon_decay : float,
            final_epsilon : float,
            discount_factor : float =  0.95,
            n_bins = (10,10),
            n_tilings = 8
    ):
        
        self.env = env

        low = env.observation_space.low
        high = env.observation_space.high
        
        self.tilecoder = TileCoder(low, high, n_bins, n_tilings)
        self.tiled_q_table = TiledQTable(tilecoder=self.tilecoder, n_actions=env.action_space.n)

        # self.q_values = np.zeros(n_bins + (env.action_space.n,))

        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor

        # self.training_error = []

    # def discretize(self, observation):
    #     return tuple(np.digitize(observation[i], self.bins[i]) for i in range(len(observation)))
    
    
    def get_action(
            self,
            observation
    ) -> int:
        
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.tiled_q_table.get_greedy_action(observation)
        
    def update(
            self, 
            observation,
            action : int,
            reward : float,
            terminated : bool,
            next_observation
    ):
        
        future_best_action = self.tiled_q_table.get_greedy_action(next_observation)

        future_q_value = (not terminated)*(self.tiled_q_table.get(next_observation, future_best_action))

        target = reward + self.discount_factor*(future_q_value)

        self.tiled_q_table.update(observation, action, target, self.lr)

        # self.training_error.append(temporal_diff)

    def e_decay(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

def main():
    print("MountainCar Q-Learning Agent")

if __name__ == "__main__":
    main()

