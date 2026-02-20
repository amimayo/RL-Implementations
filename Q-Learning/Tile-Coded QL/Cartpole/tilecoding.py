import numpy as np

# Tile Coding Implementation

# 1. TileCoder
class TileCoder:

    def __init__(
            self,
            low,
            high,
            n_bins=(10, 10, 10, 10),
            n_tilings=8
    ):
        
        self.low = np.array(low)
        self.high = np.array(high)
        self.bins = np.array(n_bins)
        self.tilings = n_tilings

        # Scale of each division in a single tile
        self.tile_width = (self.high - self.low) / (self.bins - 1) 

        # Create asymmetrical offsets to prevent diagonal stacking in 4D space
        self.offsets = []
        for i in range(self.tilings):
            dim_offsets = []
            for d in range(len(self.bins)):
                # (2d + 1) creates the staggering effect across dimensions
                shift = (i / float(self.tilings)) * self.tile_width[d] * ((2 * d + 1) % self.tilings)
                dim_offsets.append(shift)
            self.offsets.append(np.array(dim_offsets))


    def encode(self, observation):
        # Clip state as sometimes state can go out of bounds
        observation = np.clip(observation, self.low, self.high) 
        encoded = [] 
        
        for offset_idx in range(self.tilings):
            # Shift observation with the unique staggered offset for this tiling
            shifted_obs = observation + self.offsets[offset_idx] 
            indices = [] 
            
            for i in range(len(self.bins)):
                # Calculate the bin index
                index = int((shifted_obs[i] - self.low[i]) / self.tile_width[i])
                # Ensure it stays within the bounds of the array
                index = np.clip(index, 0, self.bins[i] - 1)
                indices.append(index)              
            
            # Encoded Tile Indices for the given observation
            encoded.append(tuple(indices)) 
        
        return encoded
    

# 2. TiledQTable
class TiledQTable:

    def __init__(self, tilecoder, n_actions):
        self.tilecoder = tilecoder
        self.n_actions = n_actions
        q_table_shape = (tilecoder.tilings,) + tuple(tilecoder.bins) + (n_actions,)
        self.q = np.zeros(q_table_shape) 

    def get(self, observation, action):
        encoded = self.tilecoder.encode(observation) 
        tiled_q_values = [self.q[i][obs + (action,)] for i, obs in enumerate(encoded)] 
        
        return np.sum(tiled_q_values) 

    def update(self, observation, action, target, lr=0.1):
   
        current_q_value = self.get(observation, action)
        temporal_diff = target - current_q_value 
        
        encoded = self.tilecoder.encode(observation) 
        for i, obs in enumerate(encoded):
            
            self.q[i][obs + (action,)] += lr * temporal_diff 

    def get_greedy_action(self, observation):
        q_per_action = [self.get(observation, a) for a in range(self.n_actions)] 
        return int(np.argmax(q_per_action))

if __name__ == "__main__":
    print("Tile Coding Implementation Ready")