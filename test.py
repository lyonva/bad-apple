import click
import ast
import warnings
import os
import numpy as np
import torch as th
import pandas as pd
import gymnasium as gym
import time
import pickle

from math import ceil
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper, ReseedWrapper
from stable_baselines3.common.env_util import make_vec_env
from src.env.subproc_vec_env import CustomSubprocVecEnv
from src.env.minigrid_envs import *
from src.env.safety_constraints import MiniGridSafetyCostWrapper

from src.algo.ppo_model import PPOModel
from src.algo.ppo_trainer import PPOTrainer
from src.algo.ppo_rollout import PPORollout
from src.utils.configs import TrainingConfig
from stable_baselines3.common.utils import set_random_seed, obs_as_tensor
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape

from analysis.utils import import_config_module, get_map_snaps

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
        self.costs = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)

        self.positions = np.zeros((buffer_size, n_envs, 2), dtype=np.float32)
    
    def add(self, iter, obs, actions, new_obs, ext_rewards, int_rewards, costs, dones, positions):
        self.obs[iter] = np.array(obs).copy()
        self.actions[iter] = np.array(actions).copy()
        self.new_obs[iter] = np.array(new_obs).copy()
        self.ext_rewards[iter] = np.array(ext_rewards).copy()
        self.int_rewards[iter] = np.array(int_rewards).copy()
        self.costs[iter] = np.array(costs).copy()
        self.dones[iter] = np.array(dones).copy()
        self.positions[iter] = np.array(positions).copy()

def test(config_file, models_dir, total_steps, render, deterministic):
    config = import_config_module(config_file)
    # Get all methods
    techs = config.im
    seeds = config.seeds
    maps = config.maps
    maps_name = config.maps_name
    pos = []

    for map, map_name in zip(maps, maps_name):

        model_snapshots = get_map_snaps(map_name, config.maps_snapshot, config.default_snaps)

        # buffer_size = ceil(total_steps / config.num_processes)
        # data, stats = dict( (tech, {}) for tech in techs ), dict( (tech, {}) for tech in techs )


        for snap in model_snapshots:
            # for tech in techs: data[tech][snap], stats[tech][snap] = {}, {}
            # Data collection
            # Per model
            for tech in techs:
                for seed in seeds:
                    pic = False
                    if render: print(f"{map}-{tech}-{snap}")
                    with open(os.path.join(models_dir, map, f"{tech}-{seed}", "params.pkl"), 'rb') as f:
                        config_orig = pickle.load(f)
                    config_orig["deterministic"] = bool(deterministic)
                    config_env = TrainingConfig()
                    for k, v in config_orig.items(): setattr(config_env, k, v)
                    config_env.init_env_name(config_env.game_name, None)

                    wrapper_class = config_env.get_wrapper_class()
                    venv = config_env.get_venv(wrapper_class)
                    # callbacks = config_env.get_callbacks()
                    obs = venv.set_seeds([seed for _ in range(config_env.num_processes)])

                    model = PPOTrainer.load( os.path.join(models_dir, map, f"{tech}-{seed}", f"snapshot-{snap}"), env=None )
                    model.policy.eval()

                    obs_shape = get_obs_shape(venv.observation_space)
                    action_dim = get_action_dim(venv.action_space)

                    # record = TestingRecord(buffer_size, config.num_processes, obs_shape, action_dim)
                    
                    venv = model._wrap_env(venv)

                    obs = venv.reset()
                    # if config.fixed_seed != -1:
                    #     np.random.seed(config.fixed_seed)
                    # env.waiting = True
                    # for i in range(config.num_processes):
                    #     env.send_reset(env_id=i)
                    # for i in range(config.num_processes):
                    #     obs[i] = env.recv_obs(env_id=i)
                    # env.waiting = False

                    steps = 0
                    iters = 0

                    while steps <= total_steps:
                        if not(pic):
                            img = venv.render(mode="rgb_array")
                            from PIL import Image
                            im = Image.fromarray(img)
                            im.save(f"{config_env.env_name}-{seed}.png")
                            pic = True

                        if render:
                            v.render(mode="human")
                            time.sleep(0.1)
                            

                        obs_tensor = obs_as_tensor(obs, model.device)
                        model._last_policy_mems = model._last_policy_mems.to(model.device)
                        actions, ext_values, int_values, log_probs, policy_mems = model.policy.forward( obs_tensor, model._last_policy_mems, deterministic=deterministic )
                        actions = actions.cpu().numpy()
                        if isinstance(venv.action_space, gym.spaces.Box):
                            actions = np.clip(actions, venv.action_space.low, venv.action_space.high)
                        new_obs, rewards, costs, dones, infos = venv.step(actions)
                        

                        intrinsic_rewards, model_mems = \
                            model.create_intrinsic_rewards(new_obs, actions, dones)

                        if policy_mems is not None:
                            model._last_policy_mems = policy_mems.detach().clone()
                        if model_mems is not None:
                            model._last_model_mems = model_mems.detach().clone()
                        
                        # For done envs, reset with same seed
                        # if config.fixed_seed != -1:
                            # np.random.seed(config.fixed_seed)
                        for i, d in enumerate(dones):
                            if d:
                                venv.send_reset(env_id=i)
                                new_obs[i] = venv.recv_obs(env_id=i)

                        agent_positions = np.array(venv.get_attr('agent_pos'))
                        
                        # Record
                        # record.add(iters, obs, actions.reshape(-1, 1), new_obs, rewards, intrinsic_rewards, costs, dones, agent_positions)
                        
                        # Log positions
                        for p in range(config_env.num_processes):
                            pos.append( ( map, tech, seed, snap, iters, p, agent_positions[p][0], agent_positions[p][1], rewards[p], costs[p], dones[p]) )

                        obs = new_obs
                        iters += 1
                        steps += config_env.num_processes
                    
                    # data[tech][snap][seed] = record
    
    pos_df = pd.DataFrame.from_records(pos, columns=["map", "im", "seed", "snapshot", "iterations", "n_env", "x", "y", "reward", "cost", "done"])
    pos_df.to_csv(f"analysis/positions.csv")



@click.command()
# Testing params
@click.option('--config', default='config', type=str, help='Config file')
@click.option('--models_dir', default="analysis/logs", type=str, help='Path to the folder that contains the models')
# Process options
@click.option('--total_steps', default=int(10000), type=int, help='Total number of frames to run for testing')
@click.option('--render', default=0, type=int, help="Activate rendering. Will significantly slow down the process.")
@click.option('--deterministic', default=1, type=int, help="Whether to use the most likely action instead of PPO random action sampling. 0 = Stochastic/Random, 1 = Deterministic")

def main(
    config, models_dir, total_steps, render, deterministic,
):
    test(config, models_dir, total_steps, render, deterministic)

#  python .\test.py --game_name=DoorKey-8x8 --models_dir=analysis\logs\MiniGrid-DoorKey-8x8-v0 --baseline=nors+nomodel --snaps=[500,1000,2500,5000,10000] --fixed_seed=1
if __name__ == '__main__':
    main()