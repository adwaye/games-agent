from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Game2048Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        reward_function: str = 'full_merge',
        reward_scaling: str = 'total',
    ):
        """Games env for 2048 game

        Parameters
        ----------
        reward_function: str
            Types:
            * ``'merge_only``: +1 when merge happens

            *``'full_merge'``: total of the number of merged cells

        reward_scaling: str
            Types
            * ``'total_value``: scales the reward by the total board value

            * ``'total_steps``: scales the reward by the total steps taken

            * ``'current'``: no scaling apply


        """
        super().__init__()
        self.size = 4
        self.reward_function = reward_function
        # 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.action_dict = {
            0: 'up',
            1: 'down',
            2: 'left',
            3: 'right',
        }
        self.observation_space = spaces.Box(
            low=0, high=2**16, shape=(self.size, self.size), dtype=np.int32,
        )
        self.reset()

    @property
    def reward_scaling(self):
        return self._reward_scaling

    @reward_scaling.setter
    def reward_scaling(self, value):
        allowed_rewards = ['current', 'total_steps', 'total_value']
        if value.lower() not in allowed_rewards:
            raise ValueError(
                f'reward_scaling should be one of {allowed_rewards}',
            )
        self._reward_scaling = value

    def _reward_fn(self, merged_val: float) -> float:
        if self.reward_function == 'merge_only':
            move_reward = 1.
        elif self.reward_function == 'full_merge':
            move_reward = merged_val
        elif self.reward_function == 'log_full_merge':
            move_reward = np.log2(merged_val)
        return self._scale_reward(move_reward)

    def _scale_reward(self, move_reward: float) -> float:
        if self.reward_scaling == 'total_value':
            return move_reward/np.sum(self.board)
        elif self.reward_scaling == 'total_steps':
            return move_reward/self.steps
        elif self.reward_scaling == 'current':
            return move_reward
        else:
            raise NotImplementedError('Not implemented for reward_scaling')

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self._add_tile()
        self._add_tile()
        self.steps = 0
        return self.board.copy()

    def step(self, action):
        assert self.action_space.contains(action)
        # prev_board = self.board.copy()
        self.board, reward, moved = self._move(action)
        done = not self._can_move()
        if moved:
            self._add_tile()
        self.steps += 1
        return self.board.copy(), reward, done, {}

    def render(self, mode='human'):
        print(self.board)

    def _add_tile(self):
        empty = list(zip(*np.where(self.board == 0)))
        if empty:
            y, x = random.choice(empty)
            self.board[y, x] = 4 if random.random() < 0.1 else 2

    def _move(self, direction):
        def move_row_left(row):
            non_zero = row[row != 0]
            merged = []
            skip = False
            reward = 0
            for i in range(len(non_zero)):
                if skip:
                    skip = False
                    continue
                if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                    merged_val = non_zero[i] * 2
                    merged.append(merged_val)
                    reward += self._reward_fn(merged_val)
                    skip = True
                else:
                    merged.append(non_zero[i])
            merged += [0] * (self.size - len(merged))
            return np.array(merged), reward

        board = self.board.copy()
        total_reward = 0
        moved = False

        for i in range(self.size):
            if direction == 0:  # up
                row, reward = move_row_left(board[:, i])
                if not np.array_equal(board[:, i], row):
                    moved = True
                board[:, i] = row
                total_reward += reward
            elif direction == 1:  # down
                row, reward = move_row_left(board[::-1, i])
                row = row[::-1]
                if not np.array_equal(board[:, i], row):
                    moved = True
                board[:, i] = row
                total_reward += reward
            elif direction == 2:  # left
                row, reward = move_row_left(board[i, :])
                if not np.array_equal(board[i, :], row):
                    moved = True
                board[i, :] = row
                total_reward += reward
            elif direction == 3:  # right
                row, reward = move_row_left(board[i, ::-1])
                row = row[::-1]
                if not np.array_equal(board[i, :], row):
                    moved = True
                board[i, :] = row
                total_reward += reward

        return board, total_reward, moved

    def _can_move(self):
        if np.any(self.board == 0):
            return True
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j + 1]:
                    return True
                if self.board[j, i] == self.board[j + 1, i]:
                    return True
        return False
