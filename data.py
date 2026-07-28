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
    def __len__(self):
        return len(self.buffer)
    def add_exp(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor, done: torch.Tensor):
        self.buffer.append((obs, action, reward, next_obs, done))