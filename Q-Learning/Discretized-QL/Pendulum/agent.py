import gymnasium as gym
import numpy as np
from collections import defaultdict

class PendulumAgent:

    def __init__(
            self,
            env : gym.Env,
            learning_rate : float,
            initial_epsilon : float,
            epsilon_decay : float,
            final_epsilon : float,
            discount_factor : float =  0.95,
            n_bins = (10,10,10)
    ):
        
        self.env = env

        low = env.observation_space.low
        high = env.observation_space.high
    
        self.bins = [np.linspace(low[i], high[i], n_bins[i]-1) for i in range(len(n_bins))]

        # Discretized Actions
        self.discretized_actions = np.linspace(-2.0, 2.0, 5)
        self.n_actions = len(self.discretized_actions)
        
        #self.q_values = defaultdict(lambda : np.zeros(env.action_space.n))

        self.q_values = np.zeros(n_bins + (self.n_actions,))

        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor

        self.training_error = []

    def discretize(self, observation):
        return tuple(np.digitize(observation[i], self.bins[i]) for i in range(len(observation)))
    
    
    def get_action(
            self,
            observation
    ) -> int:
        
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return int(np.argmax(self.q_values[observation]))
        
    def update(
            self, 
            observation,
            action_idx : int,
            reward : float,
            terminated : bool,
            next_observation
    ):
        
        future_q_value = (not terminated)*np.max(self.q_values[next_observation])

        target = reward + self.discount_factor*(future_q_value)

        temporal_diff = target - self.q_values[observation + (action_idx,)]

        self.q_values[observation + (action_idx,)] += self.lr*temporal_diff

        # self.training_error.append(temporal_diff)

    def e_decay(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

def main():
    print("Cartpole Q-Learning Agent")

if __name__ == "__main__":
    main()

