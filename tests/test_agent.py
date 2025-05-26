from __future__ import annotations
from unittest import TestCase

import random
import numpy as np
from games_agent.agent import DQN, ReplayMemory
from games_agent.environment.board import Game2048Env
import torch

print("Importing DQN from games_agent.agent")


class TestDqn(TestCase):
    def setUp(self):
        self.n_actions = 4  # Example number of actions
        self.model = DQN(self.n_actions)

    def test_forward_shape(self):
        input_tensor = torch.tensor(np.random.rand(1, 4, 4), dtype=torch.float32)
        output = self.model(input_tensor)
        self.assertEqual(len(output.shape), 2)
        self.assertEqual(output.shape[0], 1)  # Batch size
        # self.assertEqual(output.shape[1], 2)
        # self.assertEqual(output.shape[2], 2)
        # self.assertEqual(output.shape[2], 2)
        self.assertEqual(output.shape[1], self.n_actions)
        # self.assertEqual(output.shape[2], self.n_actions)
        # self.assertEqual(output.shape[1], self.n_actions)  # Number of actions


class TestReplayMemory(TestCase):
    def setUp(self):
        self.capacity = 10
        self.memory = ReplayMemory(self.capacity)
        self.env = Game2048Env("log_full_merge")

    def test_push(self):
        # Push some transitions to the memory
        state = self.env.reset()
        for i in range(2):
            action = 0
            observation, reward, terminated, done = self.env.step(action)

            done = False  # Example done flag
            self.memory.push(state, action, observation, reward, done)

            print(self.memory.memory)
            np.testing.assert_array_equal(self.memory.memory[i].state, state)
            np.testing.assert_array_equal(self.memory.memory[i].next_state, observation)
            np.testing.assert_array_equal(self.memory.memory[i].reward, reward)
            np.testing.assert_array_equal(self.memory.memory[i].done, done)

    def test_buffer(self):
        state = self.env.reset()
        # state = first_state
        for i in range(11):
            action = 0
            observation, reward, terminated, _ = self.env.step(action)
            if i == 1:
                first_state = state
                first_observation = observation
                first_reward = reward
                first_done = terminated
                first_action = action

            # done = False  # Example done flag
            self.memory.push(state, action, observation, reward, terminated)

            # print(self.memory.memory)
            print(i)
            if i == 10:
                np.testing.assert_array_equal(self.memory.memory[9].state, state)
                np.testing.assert_array_equal(
                    self.memory.memory[9].next_state, observation
                )
                np.testing.assert_array_equal(self.memory.memory[9].reward, reward)
                np.testing.assert_array_equal(self.memory.memory[9].done, terminated)

                np.testing.assert_array_equal(self.memory.memory[0].state, first_state)
                np.testing.assert_array_equal(
                    self.memory.memory[0].next_state, first_observation
                )
                np.testing.assert_array_equal(
                    self.memory.memory[0].reward, first_reward
                )
                np.testing.assert_array_equal(self.memory.memory[0].done, first_done)
            else:
                np.testing.assert_array_equal(self.memory.memory[i].state, state)
                np.testing.assert_array_equal(
                    self.memory.memory[i].next_state, observation
                )
                np.testing.assert_array_equal(self.memory.memory[i].reward, reward)
                np.testing.assert_array_equal(self.memory.memory[i].done, terminated)
                state = observation


if __name__ == "__main__":
    import unittest

    unittest.main()
