import torch
import numpy as np
import gymnasium as gym
from collections import deque
from model import DeepQNetwork

gamma = 0.9
lr = 0.001
min_experiences = 1_000

replay_buffer = deque([], maxlen=100_000)

network = DeepQNetwork()
target_network = DeepQNetwork()

env = gym.make("CartPole-v1", render_mode=None)

optimizer = torch.optim.Adam(network.parameters(), lr)
criterion = torch.nn.MSELoss()

next_obs, info = env.reset()
terminated, truncated = False, False

for _ in range(min_experiences):
    obs = next_obs
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    replay_buffer.append()
    if done:
        next_obs, info = env.reset()

while not (terminated or truncated):
    output = network(torch.tensor(obs))
    obs, reward, terminated, truncated, info = env.step(output.numpy())
    y = reward + gamma * torch.max(target_network(torch.tensor(obs))).float()
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()