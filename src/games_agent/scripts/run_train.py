from torch.utils.tensorboard import SummaryWriter
import hydra
import logging
from hydra.utils import instantiate
from games_agent.agent import DQNAgent
import os
import torch
from itertools import count
# from hydra.core.config_store import config_store

@hydra.main(version_base=None,config_name='training_config.yaml')
def main(cfg):
    # BATCH_SIZE = cfg.train.batch_size
    # GAMMA = cfg.train.gamma
    # EPS_START = cfg.train.eps_start
    # EPS_END = cfg.train.eps_end
    # EPS_DECAY = cfg.train.eps_decay
    TAU = cfg.agent.tau
    # LR = cfg.train.lr

    logging.info(cfg)
    output_dir = './'
    writer = SummaryWriter(output_dir)
    os.makedirs('MODELS',exist_ok=True)
    num_episodes = cfg.agent.num_episodes
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    else:
        device = torch.device("cpu")
        
    # env = instantiate(cfg.env)
    dqn_agent = instantiate(cfg.agent)

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


if __name__ == "__main__":
    main()