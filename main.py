from __future__ import annotations

import time

import torch
from games_agent.environment.board import Game2048Env


def main():
    print('Hello from games-agent!')
    print(f'cuda is available: {torch.cuda.is_available()}')
    env = Game2048Env()
    for i in range(1):
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            _, reward, done, _ = env.step(action)
            print(f"Action: {action}, Reward: {reward}")
            env.render()
            time.sleep(2)


if __name__ == '__main__':
    main()
