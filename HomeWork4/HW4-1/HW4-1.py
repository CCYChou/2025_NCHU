# hw4_1_naive_dqn_static.py

import os
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from Gridworld import Gridworld


def get_state(game):
    """
    从 game.board.components 中读取 Player/Goal/Pit/Wall 的位置，
    返回一个形状为 (size, size, 4) 的 numpy array，
    通道顺序为 [Player, Goal, Pit, Wall] 的 one-hot 编码。
    """
    size = game.board.size
    channels = []
    for piece in ['Player', 'Goal', 'Pit', 'Wall']:
        arr = np.zeros((size, size), dtype=np.float32)
        pos = game.board.components[piece].pos
        arr[pos[0], pos[1]] = 1.0
        channels.append(arr)
    return np.stack(channels, axis=-1)  # (size, size, 4)


class QNet(nn.Module):
    """Simple 2-layer MLP Q-Network."""
    def __init__(self, state_dim=64, hidden=150, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """A simple FIFO experience replay buffer."""
    def __init__(self, capacity=5000):
        self.buffer = []
        self.capacity = capacity

    def push(self, transition):
        # transition = (state, action, reward, next_state, done)
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def train_naive_dqn(
    num_episodes=1000,
    batch_size=64,
    gamma=0.99,
    lr=1e-3,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=500,
    buffer_capacity=5000
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = QNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_capacity)

    losses = []
    all_rewards = []

    for epi in range(1, num_episodes + 1):
        game = Gridworld(mode='static', size=4)
        state = torch.tensor(
            get_state(game).reshape(-1),
            dtype=torch.float32,
            device=device
        )
        done = False
        episode_reward = 0.0

        eps = eps_end + (eps_start - eps_end) * np.exp(-1.0 * epi / eps_decay)

        while not done:
            # ε-greedy
            if random.random() < eps:
                action = random.randrange(4)
            else:
                with torch.no_grad():
                    action = net(state.unsqueeze(0)).argmax(dim=1).item()

            # take action
            move_map = ['d', 'r', 'l', 'u']
            game.makeMove(move_map[action])
            reward = game.reward()
            # done when hitting pit or reaching goal
            done = (reward != -1)
            episode_reward += reward

            next_state = torch.tensor(
                get_state(game).reshape(-1),
                dtype=torch.float32,
                device=device
            )

            buffer.push((state, action, reward, next_state, done))
            state = next_state

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                s, a_b, r_b, ns, d = zip(*batch)

                s   = torch.stack(s)
                ns  = torch.stack(ns)
                a_b = torch.tensor(a_b, dtype=torch.long, device=device)
                r_b = torch.tensor(r_b, dtype=torch.float32, device=device)
                d   = torch.tensor(d,   dtype=torch.float32, device=device)

                q_pred = net(s).gather(1, a_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    q_next = net(ns).max(dim=1)[0]
                    q_target = r_b + gamma * q_next * (1.0 - d)

                loss = nn.MSELoss()(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

        all_rewards.append(episode_reward)

        if epi % 50 == 0:
            avg_r = sum(all_rewards[-50:]) / 50
            print(f"Episode {epi}/{num_episodes} | AvgReward (last50): {avg_r:.2f} | ε: {eps:.3f}")

    # Save results
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Loss over updates")
    plt.plot(losses)
    plt.xlabel("Update step")
    plt.ylabel("MSE Loss")

    plt.subplot(1, 2, 2)
    plt.title("Episode reward")
    plt.plot(all_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.tight_layout()
    plt.savefig("results/hw4_1_training.png")
    plt.show()

    torch.save(net.state_dict(), "results/naive_dqn_static.pth")
    print("Training finished. Model and plots saved to ./results/")


if __name__ == "__main__":
    train_naive_dqn()

