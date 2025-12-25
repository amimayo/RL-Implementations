import gymnasium as gym
import numpy as np
from collections import defaultdict

class TaxiAgent:

    def __init__(
            self,
            env : gym.Env,
            learning_rate : float,
            initial_epsilon : float,
            epsilon_decay : float,
            final_epsilon : float,
            discount_factor : float =  0.95
    ):
        
        self.env = env
        self.q_values = defaultdict(lambda : np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor

        self.training_error = []

    def get_action(
            self,
            observation : int
    ) -> int:
        
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_values[observation])
        
    def update(
            self, 
            observation : int,
            action : int,
            reward : float,
            terminated : bool,
            next_observation : int
    ):
        
        future_q_value = (not terminated)*np.max((self.q_values[next_observation]))

        target = reward + self.discount_factor*(future_q_value)

        temporal_diff = target - self.q_values[observation][action]

        self.q_values[observation][action] += self.lr*temporal_diff

        self.training_error.append(temporal_diff)

    def e_decay(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

def main():
    print("Taxi Q-Learning Agent")

if __name__ == "__main__":
    main()

