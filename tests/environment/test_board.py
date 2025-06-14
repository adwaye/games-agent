from __future__ import annotations

import random
from unittest import TestCase

import numpy as np
from games_agent.environment.board import Game2048Env
from parameterized import parameterized


class TestGame2048Env(TestCase):
    def setUp(self):
        self.env = Game2048Env()

    @parameterized.expand(
        (
            [
                np.array(
                    [
                        [2, 2, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ), 3, 4,
            ],
            [
                np.array(
                    [
                        [2, 2, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ), 1, 0,
            ],
            [
                np.array(
                    [
                        [2, 2, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ), 2, 4,
            ],
            [
                np.array(
                    [
                        [2, 2, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ), 0, 0,
            ],
            [
                np.array(
                    [
                        [2, 8, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ), 3, 0,
            ],
            [
                np.array(
                    [
                        [2, 8, 0, 0],
                        [0, 2, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ), 1, 0,
            ],
            [
                np.array(
                    [
                        [2, 8, 0, 0],
                        [4, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ), 2, 0,
            ],
            [
                np.array(
                    [
                        [2, 8, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ), 0, 0,
            ],
            [
                np.array(
                    [
                        [8, 8, 0, 0],
                        [0, 0, 16, 0],
                        [8, 0, 16, 0],
                        [4, 0, 0, 0],
                    ],
                ), 0, 48,
            ],
        ),
    )
    def test_step_one_step_full_merge(self, board, action, expected_reward):
        env = Game2048Env()
        env.board = board
        # setting the random seed...
        np.random.seed(42)
        random.seed(42)
        # action = 3
        print(f'action = {env.action_dict[action]}')
        env.render()
        _, reward, done, _ = env.step(action)
        # env.render()
        print(env.board)

        assert reward == expected_reward

    @parameterized.expand(
        (
            [
                np.array(
                    [
                        [2, 2, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ), 3, 1,
            ],
            [
                np.array(
                    [
                        [2, 2, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ), 1, 0,
            ],
            [
                np.array(
                    [
                        [2, 2, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ), 2, 1,
            ],
            [
                np.array(
                    [
                        [2, 2, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ), 0, 0,
            ],
            [
                np.array(
                    [
                        [2, 8, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ), 3, 0,
            ],
            [
                np.array(
                    [
                        [2, 8, 0, 0],
                        [0, 2, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ), 1, 0,
            ],
            [
                np.array(
                    [
                        [2, 8, 0, 0],
                        [4, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ), 2, 0,
            ],
            [
                np.array(
                    [
                        [2, 8, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ), 0, 0,
            ],
            [
                np.array(
                    [
                        [8, 8, 0, 0],
                        [0, 0, 16, 0],
                        [0, 0, 16, 1024],
                        [0, 0, 0, 1024],
                    ],
                ), 0, 2,
            ],
        ),
    )
    def test_step_one_step_merge_only(self, board, action, expected_reward):
        env = Game2048Env('merge_only')
        env.board = board
        # setting the random seed...
        np.random.seed(42)
        random.seed(42)
        # action = 3
        print(f'action = {env.action_dict[action]}')
        env.render()
        _, reward, done, _ = env.step(action)
        # env.render()
        print(env.board)

        assert reward == expected_reward

    def test_board_action_len(self):
        env = Game2048Env()
        assert len(
            env.action_dict,
        ) == env.action_space.n, \
            'Action space size does not match action dictionary size'
        assert env.action_space.n == 4, \
            'Action space should have 4 actions (up, down, left, right)'
