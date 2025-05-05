import click
import ast
import warnings
import os
import numpy as np
import torch as th
import pandas as pd
import gymnasium as gym

from math import ceil
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper, ReseedWrapper
from stable_baselines3.common.env_util import make_vec_env
from src.env.subproc_vec_env import CustomSubprocVecEnv
from src.env.minigrid_envs import *

from src.algo.ppo_model import PPOModel
from src.algo.ppo_trainer import PPOTrainer
from src.algo.ppo_rollout import PPORollout
from src.utils.configs import TrainingConfig
from stable_baselines3.common.utils import set_random_seed, obs_as_tensor
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape

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
    buffer_size = ceil(config.total_steps / config.num_processes)

    # Get all methods
    techs = set([x.split("-")[0] for x in os.listdir(config.models_dir)])
    seeds = config.seeds
    if len(seeds) == 0: seeds = set([int(x.split("-")[1]) for x in os.listdir(config.models_dir)])
    data, stats = dict( (tech, {}) for tech in techs ), dict( (tech, {}) for tech in techs )
    pos = []

    for snap in config.snaps:
        for tech in techs: data[tech][snap], stats[tech][snap] = {}, {}

        for seed in seeds:
            env = make_vec_env(config.env_name,
                               wrapper_class=lambda x: ImgObsWrapper(x),
                               vec_env_cls=CustomSubprocVecEnv,
                               n_envs=config.num_processes,
                            )
            model_baseline = PPOTrainer.load( os.path.join(config.models_dir, f"{config.baseline}-{seed}", f"snapshot-{snap}"), env=None )

            env = model_baseline._wrap_env(env)
            obs = env.reset()

            env.can_see_walls = model_baseline.can_see_walls
            env.image_noise_scale = model_baseline.image_noise_scale
            env.image_rng = np.random.default_rng(seed=seed + 1313)
            np.random.seed(seed)
            local_seeds = np.random.rand(config.num_processes)
            local_seeds = [int(s * 0x7fffffff) for s in local_seeds]
            np.random.seed(seed)
            env.set_seeds(local_seeds)
            env.waiting = True
            for i in range(config.num_processes):
                env.send_reset(env_id=i)
            for i in range(config.num_processes):
                obs[i] = env.recv_obs(env_id=i)
            env.waiting = False

            obs_shape = get_obs_shape(env.observation_space)
            action_dim = get_action_dim(env.action_space)

            # Data collection
            # Per model
            for tech in techs:
                model = PPOTrainer.load( os.path.join(config.models_dir, f"{tech}-{seed}", f"snapshot-{snap}"), env=None )
                record = TestingRecord(buffer_size, config.num_processes, obs_shape, action_dim)
                
                obs = env.reset()
                steps = 0
                iters = 0
                model.policy.eval()

                while steps <= config.total_steps:
                    obs_tensor = obs_as_tensor(obs, config.device)
                    actions, values, log_probs, policy_mems = model.policy.forward( obs_tensor, model._last_policy_mems )
                    actions = actions.cpu().numpy()
                    if isinstance(env.action_space, gym.spaces.Box):
                        actions = np.clip(actions, env.action_space.low, env.action_space.high)
                    new_obs, rewards, dones, infos = env.step(actions)
                    agent_positions = np.array(env.get_attr('agent_pos'))

                    intrinsic_rewards, model_mems = \
                        model.create_intrinsic_rewards(new_obs, actions, dones)

                    if policy_mems is not None:
                        model._last_policy_mems = policy_mems.detach().clone()
                    if model_mems is not None:
                        model._last_model_mems = model_mems.detach().clone()
                    
                    # Record
                    record.add(iters, obs, actions.reshape(-1, 1), new_obs, rewards, intrinsic_rewards, dones, agent_positions)
                    
                    # Log positions
                    for p in range(config.num_processes):
                        pos.append( ( tech, seed, snap, iters, p, agent_positions[p][0], agent_positions[p][1]) )

                    obs = new_obs
                    iters += 1
                    steps += config.num_processes
                
                data[tech][snap][seed] = record
    
    pos_df = pd.DataFrame.from_records(pos, columns=["im", "seed", "snapshot", "iterations", "n_env", "x", "y"])
    pos_df.to_csv(f"analysis/positions-{config.game_name}.csv")



@click.command()
# Testing params
@click.option('--game_name', default="DoorKey-8x8", type=str, help='e.g. DoorKey-8x8, FourRooms, RedBlueDoors-8x8')
# Both directories should have the same structure
@click.option('--models_dir', type=str, help='Path to the folder that contains the models')
@click.option('--baseline', default="nomodel", type=str, help='Name of the baseline method')
@click.option('--seeds', default='[]', cls=PythonLiteralOption, help='List of seeds')
@click.option('--snaps', default='[]', cls=PythonLiteralOption, help='List of model snapshots')
# Process options
@click.option('--num_processes', default=16, type=int, help='Number of testing processes (workers)')
@click.option('--total_steps', default=int(5000), type=int, help='Total number of frames to run for testing')

def main(
    game_name, models_dir, baseline, seeds, snaps, num_processes, total_steps,
):
    args = locals().items()
    config = TrainingConfig()
    for k, v in args: setattr(config, k, v)
    config.env_source = "minigrid"

    config.init_env_name(game_name, None)

    test(config)

#  python .\test.py --game_name=DoorKey-16x16 --models_dir=analysis\logs\MiniGrid-DoorKey-16x16-v0 --baseline=nomodel --snaps=[3,15,305,610,915,1221]
if __name__ == '__main__':
    main()