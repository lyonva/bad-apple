import click
import glob
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as th
import gymnasium as gym

import minigrid
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper, ReseedWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed, obs_as_tensor
from src.env.subproc_vec_env import CustomSubprocVecEnv
from src.algo.ppo_trainer import PPOTrainer

def make_policy_map(game_name, model_path):
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    env = gym.make(game_name).unwrapped
    model = PPOTrainer.load( model_path, env=None )

    # env = model._wrap_env(env)
    obs = env.reset()
    
    env.image_rng = np.random.default_rng(seed=model.seed + 1313)
    np.random.seed(model.seed)
    local_seeds = np.random.rand()
    local_seeds = int(local_seeds * 0x7fffffff)
    np.random.seed(model.seed)
    env.reset(seed=local_seeds)

    width, height = env.width, env.height
    policy = np.zeros((width, height, 4))

    for i in range(width):
        for j in range(height):
            cell = env.grid.get(i, j)
            
            if cell is None or cell.can_overlap():
                env.agent_pos = (i, j) # Manually set position
                for k in range(4):
                    env.agent_dir = k # Manually set orientation
                    obs = env.gen_obs()["image"]
                    print(obs)
                    obs_tensor = obs_as_tensor(np.array([obs]), device)

                    actions, _, _, _ = model.policy.forward( obs_tensor, model._last_policy_mems )
                    actions = actions.cpu().numpy()
                    if isinstance(env.action_space, gym.spaces.Box):
                        actions = np.clip(actions, env.action_space.low, env.action_space.high)
                    policy[i][j][k] = actions[0]

            else:
                policy[i][j] = [-1, -1, -1, -1]
    
    print(policy)


@click.command()
# Testing params
@click.option('--game_name', default="DoorKey-8x8", type=str, help='e.g. DoorKey-8x8, FourRooms, RedBlueDoors-8x8')
@click.option('--model_path', default=".", type=str, help='Directory with the model files')

def main(
    game_name, model_path,
):
    make_policy_map(game_name, model_path)
    

if __name__ == '__main__':
    main()
