from dataclasses import dataclass
from collections import deque

@dataclass
class TrainingConfig:
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_episodes: int = 2000
    learning_rate: float = 1e-4
    clipnorm: float = 1.0
    memory_size: int = 200000
    batch_size: int = 128
    target_update_freq: int = 2000
    min_memory_for_training: int = 5000
    grid_size: int = 12
    num_episodes: int = 2000
    opponent_switch_episode: int = 5
