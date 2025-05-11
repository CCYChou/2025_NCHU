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
    return np.stack(channels, axis=-1)


class QNet(nn.Module):
    """两层 MLP Q 网络"""
    def __init__(self, state_dim=64, hidden=150, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class DuelingQNet(nn.Module):
    """Dueling DQN 架构"""
    def __init__(self, state_dim=64, hidden=150, n_actions=4):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, n_actions)
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value_stream(f)              # (B,1)
        a = self.adv_stream(f)                # (B,n)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


class ReplayBuffer:
    """FIFO 经验回放缓冲区"""
    def __init__(self, capacity=5000):
        self.buffer = []
        self.capacity = capacity

    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def train_double_dqn(
    num_episodes=1000,
    batch_size=64,
    gamma=0.99,
    lr=1e-3,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=500,
    buffer_capacity=5000,
    update_target_every=50
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    online_net = QNet().to(device)
    target_net = QNet().to(device)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_capacity)

    losses, rewards = [], []
    total_steps = 0

    for epi in range(1, num_episodes+1):
        game = Gridworld(mode='player', size=4)
        state = torch.tensor(get_state(game).reshape(-1), dtype=torch.float32, device=device)
        done = False
        episode_reward = 0
        eps = eps_end + (eps_start-eps_end)*np.exp(-1.0*epi/eps_decay)

        while not done:
            # ε-greedy
            if random.random() < eps:
                action = random.randrange(4)
            else:
                with torch.no_grad():
                    action = online_net(state.unsqueeze(0)).argmax(1).item()

            game.makeMove('drlu'[action])
            r = game.reward()
            done = (r != -1)
            episode_reward += r
            next_state = torch.tensor(get_state(game).reshape(-1), dtype=torch.float32, device=device)

            buffer.push((state, action, r, next_state, done))
            state = next_state

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                s,a,r_b,ns,d = zip(*batch)
                s = torch.stack(s)
                ns = torch.stack(ns)
                a = torch.tensor(a, dtype=torch.long, device=device)
                r_b = torch.tensor(r_b, dtype=torch.float32, device=device)
                d = torch.tensor(d, dtype=torch.float32, device=device)

                q_pred = online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_actions = online_net(ns).argmax(1)
                    q_next = target_net(ns).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    q_target = r_b + gamma*q_next*(1-d)

                loss = nn.MSELoss()(q_pred, q_target)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(loss.item())

                total_steps += 1
                if total_steps % update_target_every == 0:
                    target_net.load_state_dict(online_net.state_dict())

        rewards.append(episode_reward)
        if epi % 50 == 0:
            print(f"Double DQN Episode {epi}/{num_episodes} | AvgReward(last50): {np.mean(rewards[-50:]):.2f} | ε: {eps:.3f}")

    # save and plot
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.plot(losses); plt.title('Double DQN Loss'); plt.xlabel('Step'); plt.ylabel('MSE Loss')
    plt.subplot(1,2,2); plt.plot(rewards); plt.title('Double DQN Reward'); plt.xlabel('Episode'); plt.ylabel('Reward')
    plt.tight_layout(); plt.savefig('results/hw4_2_double_dqn.png'); plt.show()
    torch.save(online_net.state_dict(), 'results/double_dqn.pth')


def train_dueling_dqn(
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
    net = DuelingQNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_capacity)

    losses, rewards = [], []

    for epi in range(1, num_episodes+1):
        game = Gridworld(mode='player', size=4)
        state = torch.tensor(get_state(game).reshape(-1), dtype=torch.float32, device=device)
        done = False
        episode_reward = 0
        eps = eps_end + (eps_start-eps_end)*np.exp(-1.0*epi/eps_decay)

        while not done:
            # ε-greedy
            if random.random() < eps:
                action = random.randrange(4)
            else:
                with torch.no_grad():
                    action = net(state.unsqueeze(0)).argmax(1).item()

            game.makeMove('drlu'[action])
            r = game.reward()
            done = (r != -1)
            episode_reward += r
            next_state = torch.tensor(get_state(game).reshape(-1), dtype=torch.float32, device=device)

            buffer.push((state, action, r, next_state, done))
            state = next_state

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                s,a,r_b,ns,d = zip(*batch)
                s = torch.stack(s)
                ns = torch.stack(ns)
                a = torch.tensor(a, dtype=torch.long, device=device)
                r_b = torch.tensor(r_b, dtype=torch.float32, device=device)
                d = torch.tensor(d, dtype=torch.float32, device=device)

                q_pred = net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = net(ns).max(dim=1)[0]
                    q_target = r_b + gamma*q_next*(1-d)

                loss = nn.MSELoss()(q_pred, q_target)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(loss.item())

        rewards.append(episode_reward)
        if epi % 50 == 0:
            print(f"Dueling DQN Episode {epi}/{num_episodes} | AvgReward(last50): {np.mean(rewards[-50:]):.2f} | ε: {eps:.3f}")

    # save and plot
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.plot(losses); plt.title('Dueling DQN Loss'); plt.xlabel('Step'); plt.ylabel('MSE Loss')
    plt.subplot(1,2,2); plt.plot(rewards); plt.title('Dueling DQN Reward'); plt.xlabel('Episode'); plt.ylabel('Reward')
    plt.tight_layout(); plt.savefig('results/hw4_2_dueling_dqn.png'); plt.show()
    torch.save(net.state_dict(), 'results/dueling_dqn.pth')


if __name__ == "__main__":
    # Train Double DQN
    train_double_dqn()
    # Train Dueling DQN
    train_dueling_dqn()

