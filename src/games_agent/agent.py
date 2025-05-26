from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import os
import math
import random
# import matplotlib
# import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from games_agent.environment.board import Game2048Env

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


lass PrioritizedReplayBuffer:
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
            i = np.random.choice(len(self.buffer), p=self.priorities/sum(self.priorities))
            self.buffer[i] = (state, action, reward, next_state, done)
            self.priorities[i] = max_priority + 1

    def sample(self, batch_size):
        if len(self.buffer) < self.capacity:
            i = np.random.choice(len(self.buffer), size=batch_size)
            return [self.buffer[j] for j in i]
        else:
            i = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=self.priorities/sum(self.priorities))
            return [self.buffer[i[j]] for j in range(batch_size)]

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


class ReplayMemory(object):

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
        super(DQN, self).__init__()
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
    def __init__(self, env,device):
        self.env = env
        self.device = device
        self.n_actions = env.action_space.n
        self.policy_net = DQN(self.n_actions).to(device)
        self.target_net = DQN(self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def save_model(self, path='dqn_model.pth'):
        torch.save(self.policy_net.state_dict(), path)
        logging.info(f"Model saved to {path}")




if __name__ == "__main__":

    from datetime import datetime
    today = datetime.today()
    # current_time = date.strftime("%H:%M:%S")
    output_dir = f"./output/{today}"

    writer = SummaryWriter(output_dir)
    os.makedirs(output_dir + '/MODELS', exist_ok=True)
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 2000
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    else:
        device = torch.device("cpu")
        num_episodes = 50

    dqn_agent = DQNAgent(Game2048Env('log_full_merge'),device=device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




    for i_episode in range(num_episodes):
        reward_total = 0
        state = dqn_agent.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = dqn_agent.select_action(state) #dqn_agent.select_action(state)
            observation, reward, terminated, done = dqn_agent.env.step(action.item())
            reward_total += reward
            reward = torch.tensor([reward], device=device)
            # done = terminated or truncated

            if terminated:
                next_state = None
                done = True
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            dqn_agent.memory.push(state, action, next_state, reward)

            state = next_state

            dqn_agent.optimize_model()

            target_net_state_dict = dqn_agent.target_net.state_dict()
            policy_net_state_dict = dqn_agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            dqn_agent.target_net.load_state_dict(target_net_state_dict)
            
            if done:
                if i_episode % 10 == 0:
                    logging.info(f"Episode {i_episode} finished after {t + 1} timesteps")
                    logging.info(f'Total reward: {reward_total}')
                    logging.info(f'Max value in board: {dqn_agent.env.board.max()}')
                    dqn_agent.env.render()
                    writer.add_scalar('Episode/Reward', reward_total, i_episode)
                    writer.add_scalar('Episode/Duration', t + 1, i_episode)
                    writer.add_scalar('Episode/max_value', dqn_agent.env.board.max(), i_episode)
                if i_episode % 100 == 0:
                    dqn_agent.save_model(path = output_dir + f'/MODELS/dqn_model_{i_episode}.pth')
                    
                
                break
                
        





"""
    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)


    steps_done = 0


    def select_action(state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


    episode_durations = []


    def plot_durations(show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())


    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()


    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()
"""