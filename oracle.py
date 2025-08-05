import click
import ast
import warnings
import os
import numpy as np
import torch as th
import pandas as pd
import gymnasium as gym
import time

from math import ceil
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper, ReseedWrapper
from stable_baselines3.common.env_util import make_vec_env
from src.env.subproc_vec_env import CustomSubprocVecEnv
from src.env.minigrid_envs import *

from src.utils.configs import TrainingConfig
from src.algo.oracle_model import Oracle

warnings.filterwarnings("ignore", category=DeprecationWarning)

# From: https://stackoverflow.com/questions/47631914/how-to-pass-several-list-of-arguments-to-click-option
class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)
        

class TestingRecord:
    def __init__(self, buffer_size, n_envs, obs_shape, action_dim):
        self.obs = np.zeros((buffer_size, n_envs) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_envs, action_dim), dtype=np.float32)
        self.new_obs = np.zeros((buffer_size, n_envs) + obs_shape, dtype=np.float32)
        self.ext_rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.int_rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)

        self.positions = np.zeros((buffer_size, n_envs, 2), dtype=np.float32)
    
    def add(self, iter, obs, actions, new_obs, ext_rewards, int_rewards, dones, positions):
        self.obs[iter] = np.array(obs).copy()
        self.actions[iter] = np.array(actions).copy()
        self.new_obs[iter] = np.array(new_obs).copy()
        self.ext_rewards[iter] = np.array(ext_rewards).copy()
        self.int_rewards[iter] = np.array(int_rewards).copy()
        self.dones[iter] = np.array(dones).copy()
        self.positions[iter] = np.array(positions).copy()


def float_zeros(tensor_shape, config):
    return th.zeros(tensor_shape, device=config.device, dtype=th.float32)

def test(config):
    # buffer_size = config.total_steps

    # Get all methods
    seeds = config.seeds

    rm = "human" if config.render != 0 else None
    env = gym.make(config.env_name, render_mode=rm)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    model = Oracle(env, config.env_name)

    for seed in seeds:
        model.reset()
        pos = []

        env.can_see_walls = False
        env.image_noise_scale = 0
        
        np.random.seed(seed)
        # env.waiting = True
        # for i in range(config.num_processes):
        #     env.send_reset(env_id=i)
        # for i in range(config.num_processes):
        #     obs[i] = env.recv_obs(env_id=i)
        # env.waiting = False

        # Data collection
        # Per model
        # record = TestingRecord(buffer_size, config.num_processes, obs.shape(), env.action_space.shape[0])
        
        obs, info = env.reset(seed=seed)
        
        steps = 0
        iters = 0

        while steps <= config.total_steps:

            if config.render:
                # env.render(mode="human")
                time.sleep(0.1)
            
            action = model.forward(obs)

            new_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # For done envs, reset with same seed
            if done:
                new_obs, info = env.reset(seed=seed)
                model.reset()

            agent_positions = env.unwrapped.agent_pos
            
            # Record
            # record.add(iters, obs, actions.reshape(-1, 1), new_obs, rewards, intrinsic_rewards, dones, agent_positions)
            
            # Log positions
            pos.append( ( "oracle", seed, 0, iters, 0, agent_positions[0], agent_positions[1], reward, done) )

            obs = new_obs
            iters += 1
            steps += 1
            
            # data[tech][snap][seed] = record
    
        pos_df = pd.DataFrame.from_records(pos, columns=["im", "seed", "snapshot", "iterations", "n_env", "x", "y", "reward", "done"])
        pos_df.to_csv(f"analysis/positions-oracle-{config.game_name}{'-fixed' + str(seed)}.csv")



@click.command()
# Testing params
@click.option('--game_name', default="DoorKey-8x8", type=str, help='e.g. DoorKey-8x8, FourRooms, RedBlueDoors-8x8')
# Both directories should have the same structure
@click.option('--seeds', default='[]', cls=PythonLiteralOption, help='List of map seeds')
# Process options
@click.option('--total_steps', default=int(5000), type=int, help='Total number of frames to run for testing')
@click.option('--render', default=0, type=int, help="Activate rendering. Will significantly slow down the process.")

def main(
    game_name,  seeds, total_steps, render,
):
    args = locals().items()
    config = TrainingConfig()
    for k, v in args: setattr(config, k, v)
    config.env_source = "minigrid"

    config.init_env_name(game_name, None)

    test(config)

#  python .\test.py --game_name=DoorKey-8x8 --models_dir=analysis\logs\MiniGrid-DoorKey-8x8-v0 --baseline=nors+nomodel --snaps=[500,1000,2500,5000,10000] --fixed_seed=1
if __name__ == '__main__':
    main()