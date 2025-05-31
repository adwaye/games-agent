from __future__ import annotations

import logging
import math
import random
from collections import deque
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'),
)


class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.indexes = np.zeros(capacity, dtype=np.int32)
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        max_priority = np.max(self.priorities)
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            n = len(self.buffer) - 1
            self.priorities[n] = max_priority + 1
        else:
            i = np.random.choice(
                len(self.buffer), p=self.priorities / sum(self.priorities),
            )
            self.buffer[i] = (state, action, reward, next_state, done)
            self.priorities[i] = max_priority + 1

    def sample(self, batch_size):
        if len(self.buffer) < self.capacity:
            i = np.random.choice(len(self.buffer), size=batch_size)
            return [self.buffer[j] for j in i]
        else:
            i = np.random.choice(
                len(self.buffer),
                size=batch_size,
                replace=False,
                p=self.priorities / sum(self.priorities),
            )
            return [self.buffer[i[j]] for j in range(batch_size)]

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.layer1 = nn.Conv2d(
            in_channels=1,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = nn.Linear(8, 8)
        self.layer3 = nn.Linear(8, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.pool1(F.relu(x))
        x = F.relu(self.layer2(x.view(x.size(0), -1)))
        x = self.layer3(x)

        return x


class DQNAgent:
    def __init__(
        self,
        env,
        batch_size: int = BATCH_SIZE,
        gamma: float = GAMMA,
        eps_start: float = EPS_START,
        eps_end: float = EPS_END,
        eps_decay: float = EPS_DECAY,
        tau: float = TAU,
        lr: float = LR,
        num_episodes: int = 2000,
        update_rule: str = 'weighted_average',
        update_interval=500,
    ):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = env
        self.device = device
        self.n_actions = env.action_space.n
        self.policy_net = DQN(self.n_actions).to(device)
        self.target_net = DQN(self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=LR, amsgrad=True,
        )
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr
        self.update_rule = update_rule
        self.update_interval = update_interval

    @property
    def update_rule(self):
        return self._update_rule

    @update_rule.setter
    def update_rule(self, value):
        assert value.lower() in ['copy', 'weighted_average']
        self._update_rule = value.lower()

    def select_action(self, state, train=True):
        """Select action
        """
        if train:
            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
                -1.0 * self.steps_done / EPS_DECAY,
            )
            self.steps_done += 1
            if sample > eps_threshold:
                return self._select_predicted_action(state)

            else:
                return torch.tensor(
                    [[self.env.action_space.sample()]],
                    device=self.device,
                    dtype=torch.long,
                )
        else:
            return self._select_predicted_action(state)

    @torch.no_grad
    def _select_predicted_action(self, state):
        return self.policy_net(state).max(1).indices.view(1, 1)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None],
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(
            state_batch,
        ).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        expected_state_action_values = (
            next_state_values * GAMMA
        ) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(
            state_action_values,
            expected_state_action_values.unsqueeze(1),
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def save_model(self, path='dqn_model.pth'):
        torch.save(self.policy_net.state_dict(), path)
        logging.info(f'Model saved to {path}')

    def update_agent_weights(self):
        """Update the weights of the target network."""
        timestep = self.steps_done
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        if self.update_rule == 'weighted_average':
            logging.info(
                'transfering weights to target network using weighted average',
            )
            for key in target_net_state_dict.keys():
                target_net_state_dict[key] = (
                    self.tau * policy_net_state_dict[key]
                    + (1 - self.tau) * target_net_state_dict[key]
                )
        elif self.update_rule == 'copy':
            if timestep % self.update_interval == 0:
                logging.info(
                    f'n_steps: {timestep}, updating target network'
                    'transfering weights to target network using copy',
                )
                for key in target_net_state_dict.keys():
                    target_net_state_dict[key] = policy_net_state_dict[key]

        self.target_net.load_state_dict(target_net_state_dict)
