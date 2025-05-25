
from __future__ import annotations
from unittest import TestCase

import random
import numpy as np
from games_agent.agent import DQN
from parameterized import parameterized
print("Importing DQN from games_agent.agent")

def TestDQN(TestCase):
    def setUp(self):
        self.n_actions = 4  # Example number of actions
        self.model = DQN(self.n_actions)

    
    def test_forward_shape(self):
        input_tensor = torch.tensor(np.random.rand( 1, 4, 4), dtype=torch.float32)
        output = self.model(input_tensor)
        self.assertEqual(output.shape[0], 2)  # Batch size
        self.assertEqual(output.shape[2], 2)
        # self.assertEqual(output.shape[1], self.n_actions)  # Number of actions
        assert output.shape[0] ==2