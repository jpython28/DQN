import torch
from torch import nn

class DeepQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 2)
    def forward(self, x):
        return nn.functional.softmax(self.linear3(nn.functional.relu(self.linear2(nn.functional.relu(self.linear1(x))))), dim=-1)