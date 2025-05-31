from __future__ import annotations

import logging
import os

import hydra
import torch
from hydra.utils import instantiate


@hydra.main(version_base=None, config_name='training_config.yaml')
def main(cfg):
    # BATCH_SIZE = cfg.train.batch_size
    # GAMMA = cfg.train.gamma
    # EPS_START = cfg.train.eps_start
    # EPS_END = cfg.train.eps_end
    # EPS_DECAY = cfg.train.eps_decay
    # TAU = cfg.agent.tau
    # LR = cfg.train.lr
    model_weights_path =\
        '/home/adwaye/AI_Models/games_agent/MODELS/dqn_model_0.pth'
    logging.info(cfg)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logging.info(f'output_dir: {output_dir}')

    os.makedirs(os.path.join(output_dir, 'MODELS'), exist_ok=True)

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    else:
        device = torch.device('cpu')

    dqn_agent = instantiate(cfg.agent)
    dqn_agent.policy_net.load_state_dict(
        torch.load(
            model_weights_path, weights_only=True,
        ),
    )
    state = dqn_agent.env.reset()
    state = torch.tensor(
        state, dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    dqn_agent.env.render()

    action = dqn_agent.select_action(state, train=False).item()
    print(action)
    dqn_agent.env.step(action)
    dqn_agent.env.render()


if __name__ == '__main__':
    main()
