from __future__ import annotations

import logging
import os
from itertools import count

import hydra
import torch
from hydra.utils import instantiate
from torch.utils.tensorboard import SummaryWriter

# from hydra.core.config_store import config_store


@hydra.main(version_base=None, config_name='training_config.yaml')
def main(cfg):
    # BATCH_SIZE = cfg.train.batch_size
    # GAMMA = cfg.train.gamma
    # EPS_START = cfg.train.eps_start
    # EPS_END = cfg.train.eps_end
    # EPS_DECAY = cfg.train.eps_decay
    # TAU = cfg.agent.tau
    # LR = cfg.train.lr

    logging.info(cfg)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logging.info(f'output_dir: {output_dir}')

    writer = SummaryWriter(output_dir)
    os.makedirs(os.path.join(output_dir, 'MODELS'), exist_ok=True)
    num_episodes = cfg.agent.num_episodes
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    else:
        device = torch.device('cpu')

    # env = instantiate(cfg.env)
    dqn_agent = instantiate(cfg.agent)
    n_step = 0
    for i_episode in range(num_episodes):
        reward_total = 0
        state = dqn_agent.env.reset()
        state = torch.tensor(
            state, dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        for t in count():
            n_step += 1
            # logging.info(f"n_step: {n_step}, {i_episode}")
            # dqn_agent.select_action(state)
            action = dqn_agent.select_action(state)
            observation, reward, terminated, done = dqn_agent.env.step(
                action.item(),
            )
            reward_total += reward
            reward = torch.tensor([reward], device=device)
            # done = terminated or truncated

            if terminated:
                next_state = None
                done = True
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device,
                ).unsqueeze(0)

            dqn_agent.memory.push(
                state, action, next_state, reward, terminated,
            )

            state = next_state

            dqn_agent.optimize_model()

            dqn_agent.update_agent_weights()

            if done:
                if n_step % 10 == 0:
                    logging.info(
                        f'Episode {i_episode} finished after {t + 1} steps',
                    )
                    logging.info(f'Total reward: {reward_total}')
                    logging.info(
                        f'Max value in board: {dqn_agent.env.board.max()}',
                    )
                    dqn_agent.env.render()
                    writer.add_scalar('Episode/Reward', reward_total, n_step)
                    writer.add_scalar('Episode/Duration', t + 1, n_step)
                    writer.add_scalar(
                        'Episode/max_value', dqn_agent.env.board.max(), n_step,
                    )

                    target_net_state_dict = dqn_agent.target_net.state_dict()
                    policy_net_state_dict = dqn_agent.policy_net.state_dict()
                    for key in target_net_state_dict.keys():
                        writer.add_histogram(
                            'TargetNet/' +
                            key, target_net_state_dict[key], n_step,
                        )
                    for key in policy_net_state_dict.keys():
                        writer.add_histogram(
                            'PolicyNet/' +
                            key, policy_net_state_dict[key], n_step,
                        )

                # Save the model parameters to a file
                # torch.save(policy_net_state_dict, path)
                if i_episode % 100 == 0:
                    dqn_agent.save_model(
                        path=output_dir + f'/MODELS/dqn_model_{i_episode}.pth',
                    )

                break


if __name__ == '__main__':
    main()
