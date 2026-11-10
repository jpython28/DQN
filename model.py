import torch
from torch import nn

class DeepQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 16)
        self.linear2 = nn.Linear(16, 8)
        self.linear3 = nn.Linear(8, 2)
    def forward(self, x):
        return self.linear3(nn.functional.relu(self.linear2(nn.functional.relu(self.linear1(x)))))