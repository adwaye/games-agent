from __future__ import annotations

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
        ),
    )
    def test_step_one_step(self, board, action, reward):
        self.env.board = np.array(
            [
                [2, 2, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        )
        # setting the random seed...
        np.random.seed(42)
        action = 3
        self.env.render()
        _, reward, done, _ = self.env.step(action)
        self.env.render()
        assert reward == 4
