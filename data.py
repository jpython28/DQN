import torch
from collections import deque
import gymnasium as gym

class ReplayBuffer(torch.utils.data.Dataset):
    def __init__(self, max_exp: int):
        super().__init__()
        self.max_exp = max_exp
        self.buffer = deque([], self.max_exp)
    def __getitem__(self, idx):
        return self.buffer[idx]
    def add_exp(self, obs, action, reward, next_obs, done):
        self.buffer.append(tuple([torch.tensor(i) for i in (obs, action, reward, next_obs, done)]))