import torch
import numpy as np
import gymnasium as gym
from collections import deque
from model import DeepQNetwork
from data import ReplayBuffer
from utils import as_tensor

exploration = 1.0
exploration_decay = 0.98
min_exploration = 0.01
discount = 0.99
lr = 0.001
max_experiences = 1_000
min_experiences = 500
num_steps = 100_000
batch_size = 32
target_update_freq = 500
log_freq = 1000
replay_buffer = ReplayBuffer(max_experiences)
explore_freq = 10
exp_decay_freq = 250

network = DeepQNetwork()
target_network = DeepQNetwork()

optimizer = torch.optim.Adam(network.parameters(), lr)
criterion = torch.nn.HuberLoss()

def explore_run(exploration: float, render=False):
    was_training = network.training
    network.eval()
    exp_env = gym.make("CartPole-v1", render_mode="human" if render else None)
    exp_next_obs, _ = exp_env.reset()
    done = False
    steps = 0
    while not done:
        exp_obs = exp_next_obs
        if torch.rand(((1,))) < exploration:
            exp_action = exp_env.action_space.sample()
        else:
            exp_action = torch.argmax(network(torch.tensor(exp_obs))).detach().numpy()
        exp_next_obs, exp_reward, exp_terminated, exp_truncated, _ = exp_env.step(exp_action)
        if render:
            exp_env.render()
        done = exp_terminated or exp_truncated
        replay_buffer.add_exp(*as_tensor(exp_obs, exp_action, exp_reward, exp_next_obs if not done else torch.zeros(exp_env.observation_space.shape), done))
        steps += 1
    network.training = was_training
    return steps

while len(replay_buffer) < min_experiences:
    explore_run(1)

total_steps = 0
losses = []
exploration_steps = []

while total_steps < num_steps:
    for i, batch in enumerate(torch.utils.data.DataLoader(dataset=replay_buffer, batch_size=batch_size, shuffle=True)):
        network.train()
        optimizer.zero_grad()
        obs, action, reward, next_obs, done = batch
        output = network(obs)
        q = torch.gather(output, dim=1, index=action.unsqueeze(1)).squeeze(-1)
        with torch.no_grad():
            y = reward + discount * torch.amax(target_network(next_obs), dim=-1) * (1-done.int())
        loss = criterion(q, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        network.eval()
        if total_steps % explore_freq == 0:
            exploration_steps.append(explore_run(exploration))
        if total_steps % exp_decay_freq == 0:
            exploration = max(exploration_decay*exploration, min_exploration)
        if total_steps % target_update_freq == 0:
            target_network.load_state_dict(network.state_dict())
        if total_steps % log_freq == 0:
            print(f"[{round(float(total_steps/num_steps*100), 1)}%] loss {round(sum(losses)/len(losses), 3)} exploration {round(exploration, 3)} run len {round(sum(exploration_steps)/len(exploration_steps), 2)} exp {len(replay_buffer)}")
            losses.clear()
            exploration_steps.clear()
        total_steps += 1
        if total_steps >= num_steps:
            break


test_env = gym.make("CartPole-v1", render_mode="human")
obs, _ = test_env.reset()
while True:
    action = torch.argmax(network(torch.tensor(obs)))
    obs, reward, terminated, truncated, _ = test_env.step(action.numpy())
    test_env.render()
    if terminated or truncated:
        obs, _ = test_env.reset()