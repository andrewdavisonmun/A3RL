#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 15:21:09 2025

@author: andrewdavison
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Grid and MDP setup
n_rows, n_cols = 5, 5
n_states = n_rows * n_cols
gamma = 0.95
alpha = 0.1
epsilon = 0.1
episodes = 5000
max_steps = 100
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
n_actions = len(actions)

# State indexing helpers
def state_to_pos(s):
    return divmod(s, n_cols)

def pos_to_state(row, col):
    return row * n_cols + col

# Special states
start_state = pos_to_state(4, 0)
red_states = [pos_to_state(2, c) for c in [0, 1, 3, 4]]
black_states = [pos_to_state(0, 0), pos_to_state(0, 4)]

# Step function
def step(state, action):
    row, col = state_to_pos(state)
    d_row, d_col = actions[action]
    new_row, new_col = row + d_row, col + d_col

    # Stay in place if out of bounds
    if not (0 <= new_row < n_rows and 0 <= new_col < n_cols):
        return state, -1, False

    next_state = pos_to_state(new_row, new_col)

    if next_state in red_states:
        return start_state, -20, False
    elif next_state in black_states:
        return next_state, -1, True
    else:
        return next_state, -1, False

# ε-greedy action selection
def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    return np.argmax(Q[state])

# SARSA and Q-learning training
def train(method='sarsa'):
    Q = np.zeros((n_states, n_actions))
    rewards = []

    for ep in range(episodes):
        state = start_state
        action = epsilon_greedy(Q, state, epsilon)
        total_reward = 0

        for _ in range(max_steps):
            next_state, reward, done = step(state, action)
            next_action = epsilon_greedy(Q, next_state, epsilon)
            total_reward += reward

            if method == 'sarsa':
                target = reward + gamma * Q[next_state][next_action]
            elif method == 'qlearning':
                target = reward + gamma * np.max(Q[next_state])
            else:
                raise ValueError("Invalid method. Choose 'sarsa' or 'qlearning'.")

            Q[state][action] += alpha * (target - Q[state][action])
            state, action = next_state, next_action

            if done:
                break

        rewards.append(total_reward)

    return Q, rewards

# Trajectory following learned greedy policy
def extract_greedy_trajectory(Q):
    state = start_state
    trajectory = [state]

    for _ in range(50):
        action = np.argmax(Q[state])
        next_state, _, done = step(state, action)
        trajectory.append(next_state)
        if done:
            break
        state = next_state

    return trajectory

# Train both methods
Q_sarsa, rewards_sarsa = train('sarsa')
Q_qlearn, rewards_qlearn = train('qlearning')

# Extract greedy trajectories
traj_sarsa = extract_greedy_trajectory(Q_sarsa)
traj_qlearn = extract_greedy_trajectory(Q_qlearn)

# Plot reward per episode
plt.figure(figsize=(12, 6))
plt.plot(rewards_sarsa, label='SARSA')
plt.plot(rewards_qlearn, label='Q-learning')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print trajectory positions
def print_trajectory(traj, label):
    print(f"\n{label} trajectory:")
    for state in traj:
        r, c = state_to_pos(state)
        print(f"({r}, {c})", end=" → ")
    print("TERMINATED")

print_trajectory(traj_sarsa, "SARSA")
print_trajectory(traj_qlearn, "Q-learning")

def plot_policy(Q, title):
    policy_grid = np.argmax(Q, axis=1).reshape((n_rows, n_cols))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)

    for i in range(n_rows):
        for j in range(n_cols):
            state = pos_to_state(i, j)
            if state in red_states:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='red', alpha=0.4))
            elif state in black_states:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='black', alpha=0.6))
            elif state == start_state:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='blue', alpha=0.4))

            action = policy_grid[i, j]

            if action == 0:
                dx, dy = -0.3, 0
            elif action == 1:
                dx, dy = 0.3, 0
            elif action == 2:
                dx, dy = 0, -0.3
            elif action == 3:
                dx, dy = 0, 0.3

            if state not in black_states:
                ax.arrow(j, i, dy, dx, head_width=0.15, head_length=0.15, fc='k', ec='k')

    ax.set_title(title)
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(np.arange(n_cols))
    ax.set_yticklabels(np.arange(n_rows))
    ax.grid(True)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


plot_policy(Q_sarsa, "SARSA: Policy Only")
plot_policy(Q_qlearn, "Q-learning: Policy Only")

def plot_trajectory_arrows(traj, title):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)

    for state in red_states:
        r, c = state_to_pos(state)
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='red', alpha=0.4))
    for state in black_states:
        r, c = state_to_pos(state)
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='black', alpha=0.6))
    r, c = state_to_pos(start_state)
    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='blue', alpha=0.4))

    for i in range(len(traj) - 1):
        r1, c1 = state_to_pos(traj[i])
        r2, c2 = state_to_pos(traj[i+1])
        dx = c2 - c1
        dy = r2 - r1
        ax.arrow(c1, r1, dx * 0.8, dy * 0.8, head_width=0.2, head_length=0.2, fc='k', ec='k')

    ax.set_title(title)
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(np.arange(n_cols))
    ax.set_yticklabels(np.arange(n_rows))
    ax.grid(True)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
plot_trajectory_arrows(traj_qlearn, "Q-learning: Trajectory")
plot_trajectory_arrows(traj_sarsa, "SARSA: Trajectory")
