import gymnasium as gym
from collections import defaultdict
import numpy as np

class BlackJackAgent:
    def __init__(
       self,
       learning_rate : float,
       initial_epsilon: float,
       epsilon_decay : float,
       final_epsilon: float,
       discount_factor: float = 0.95,
       env = gym.Env

    ):
        
        self.env = env

        self.q_values = defaultdict(lambda : np.zeros(self.env.action_space.n))

        self.lr =  learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor

        self.training_error = []
        
    def get_action(
            self,
            observation : tuple[int, int, bool]                  
                ) -> int :
        
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[observation]))
        

    def update(
            
            self,
            observation : tuple[int, int, bool],
            action : float,
            reward : int,
            terminated : bool,
            next_observation : tuple[int, int, bool]
    ):
        
        future_q_value = (not terminated) * np.max(self.q_values[next_observation])

        target = reward + self.discount_factor*(future_q_value)

        temporal_diff = target - self.q_values[observation][action]

        self.q_values[observation][action] += self.lr*temporal_diff

        self.training_error.append(temporal_diff)


    def e_decay(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


def main():
    print("Blackjack Q-Learning Agent")

if __name__ == "__main__":
    main()
        