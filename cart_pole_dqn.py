import random
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import deque

class DQN(nn.Module):

    def __init__(self, state_dim = 4, actions = 2, hidden = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, actions)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:

    def __init__(self, maxlen = 10000):
        self.buffer = deque(maxlen = maxlen)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, sample_size):
        return random.sample(self.buffer, sample_size)

    def __len__(self):
        return len(self.buffer)


# Performs the forward step and optimization steps
def optimize(mini_batch, policy_dqn, target_dqn, optimizer, discount):

    # Convert the mini batch into 5 tensors of size (64,)
    states, actions, next_states, rewards, dones = zip(*mini_batch)

    # Turn them into tensors that I can actually use. States and next_states
    # have a batch dimension here.
    states = torch.cat(states)
    actions = torch.stack(actions)
    next_states = torch.cat(next_states)
    rewards = torch.stack(rewards)
    dones = torch.tensor(dones, dtype=torch.float)

    # Calculate q-values using target policy & Bellman one-step lookahead
    with torch.no_grad():
        next_best = target_dqn(next_states).max(dim=1)[0]
        targets = rewards + discount * next_best * (1 - dones)

    # Calculate q-values under current policy
    current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Calculate loss
    loss = F.mse_loss(targets, current_q)

    # Optimization steps
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Use this to run the model. Anything with default values is a hyperparameter 
# that you can set manually when you call the function
def run(env, episodes, epsilon_start=1, epsilon_min=0.01, epsilon_decay=0.995,
        batch_size=64, network_sync_rate=10, discount=0.99, lr=1e-3):

    # Initialize policy & target networks, replay buffer
    policy_dqn = DQN()
    target_dqn = DQN()
    target_dqn.load_state_dict(policy_dqn.state_dict())
    memory = ReplayBuffer()

    # Initialize our optimizer
    optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=lr)

    # Initialize rewards list so we can graph it later
    reward_history = []

    # Initialize epsilon
    epsilon = epsilon_start

    # Initialize step count for replay buffer
    step_count = 0

    for i in range(episodes):

        # Initialize stuff for the forward step
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        terminated, truncated = False, False
        total_reward = 0

        while not terminated and not truncated:

            # Choose action by being epsilon greedy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_dqn(state).argmax().item()

            # Execute action, observe reward and next state
            next_state, reward, terminated, truncated, info = env.step(action)

            # Update reward
            total_reward += reward

            # Turn everything into tensors for replay buffer, combine terminated and
            # truncated into a single boolean
            action = torch.tensor(action, dtype=torch.long)
            next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
            reward = torch.tensor(reward, dtype=torch.float)
            done = terminated or truncated

            # Store transition in replay memory
            memory.append((state, action, next_state, reward, done))

            # Draw from replay memory once enough steps have been taken
            if len(memory) > batch_size:

                mini_batch = memory.sample(batch_size)
                optimize(mini_batch, policy_dqn, target_dqn, optimizer, discount)

            # Check if I have to update target network
            if step_count % network_sync_rate == 0:
                target_dqn.load_state_dict(policy_dqn.state_dict())

            # Take care of increments
            state = next_state
            step_count += 1

        # Add total reward to our history
        reward_history.append(total_reward)

        # Decay epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    return policy_dqn, reward_history

# Train the model
env = gym.make("CartPole-v1")
network, rewards = run(env, 500)

# Test it out
env = gym.make("CartPole-v1", render_mode="human")
state, _ = env.reset()

for i in range(500):

    state = torch.tensor(state, dtype=torch.float32)
    action = network(state).argmax().item()
    state, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

# Make the plots
rewards = np.array(rewards)

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("CartPole with DQN")
plt.show()

env.close()