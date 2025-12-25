import numpy as np

#Tile Coding Implementation

#TileCoder

class TileCoder:

    def __init__(
            self,
            low,
            high,
            n_bins=(10,10),
            n_tilings = 8
    ):
        
        self.low = np.array(low)
        self.high = np.array(high)
        self.bins= np.array(n_bins)
        self.tilings = n_tilings

        self.tile_width = (self.high - self.low)/(self.bins - 1) #Scale of each division in a single tile

        self.offsets = [(i/self.tilings)*(self.tile_width/2) for i in range(self.tilings)] #Hardcoded offsets for each tile

    def encode(self, observation):
        observation = np.clip(observation, self.low, self.high) #Clip state as sometimes state can go out of bounds
        encoded = [] 
        for offset in self.offsets:
            shifted_obs = observation + offset #Shift observation with appropriate offset
            indices = [] #Tile indices
            for i in range(len(self.bins)):
                index = int((shifted_obs[i] - self.low[i]) / self.tile_width[i])
                index = np.clip(index, 0, self.bins[i] - 1)
                indices.append(index)              
            encoded.append(tuple(indices)) #Encoded Tile Indices for given observation
        
        return encoded
    
#TiledQTable

class TiledQTable:

    def __init__(
            self,
            tilecoder,
            n_actions
    ):
        
        self.tilecoder = tilecoder
        self.n_actions = n_actions

        self.q = np.zeros((tilecoder.tilings, *tilecoder.bins, n_actions)) #Tiled Q-Table with shape = (n_tilings,n_bins[0],n_bins[1],n_actions)
        #Feature vector size = n_tilings*n_bins[0]*n_bins[1]*n_actions
    def get(self,observation,action):

        encoded = self.tilecoder.encode(observation) #Encode observation
        tiled_q_values = [self.q[i][obs + (action,)] for i, obs in enumerate(encoded)] #Get q_values for each tile
        return np.mean(tiled_q_values) #Mean of q-values for given (state,action) pair for each tile

    def update(self,observation, action, target, lr=0.1):

        encoded = self.tilecoder.encode(observation) 
        for i, obs in enumerate(encoded):
            old_value = self.q[i][obs + (action,)] #Older q_value
            temporal_diff = target - old_value #Temporal Difference Error
            self.q[i][obs + (action,)] += lr*(temporal_diff) #Bellman's Equation

    def get_greedy_action(self, observation):

        q_per_action = [self.get(observation, a) for a in range(self.n_actions)] #Get mean q_value for given observation for each action
        return int(np.argmax(q_per_action)) #Return action with maximum mean q_value i.e. greedy

if __name__ == "__main__":
    print("Tile Coding Implementation")