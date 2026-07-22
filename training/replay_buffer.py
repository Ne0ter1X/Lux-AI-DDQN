import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, maxlen=200000):
        self.buffer = deque(maxlen=maxlen)

    def push(self, state, action_map, reward, next_state, done, next_mask):
        self.buffer.append((state, action_map, reward, next_state, done, next_mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([b[0] for b in batch])
        action_maps = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        next_masks = np.array([b[5] for b in batch])
        return states, action_maps, rewards, next_states, dones, next_masks

    def __len__(self):
        return len(self.buffer)
