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
from src.utils.video_recorder import VecVideoRecorder

from src.algo.ppo_model import PPOModel
from src.algo.ppo_trainer import PPOTrainer
from src.algo.ppo_rollout import PPORollout
from src.utils.configs import TrainingConfig
from stable_baselines3.common.utils import set_random_seed, obs_as_tensor
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape

warnings.filterwarnings("ignore", category=DeprecationWarning)  


def make_video(config):
    th.autograd.set_detect_anomaly(False)
    th.set_default_dtype(th.float32)
    th.backends.cudnn.benchmark = False

    # Load original configuration
    config_path = os.path.join(os.path.dirname(config.model), "params.pkl")
    with open(config_path, 'rb') as f:
        config_orig = pickle.load(f)
    original_processes = config_orig["num_processes"]
    config_orig["num_processes"] = 1
    config_orig["deterministic"] = bool(config.deterministic)
    config_env = TrainingConfig()
    for k, v in config_orig.items(): setattr(config_env, k, v)

    config_env.init_env_name(config_env.game_name, None)

    wrapper_class = config_env.get_wrapper_class()
    venv = config_env.get_venv(wrapper_class)
    # callbacks = config_env.get_callbacks()
    obs = venv.set_seeds([config.seed])

    model = PPOTrainer.load( config.model, env=None )
    model.policy.eval()
    

    vid_name = f"{config_env.game_name}-{os.path.basename(os.path.dirname(config.model))}"
    venv = VecVideoRecorder(venv, video_folder="analysis/",
        record_video_trigger=lambda x: True,
        video_length=config.total_steps,
        name_prefix=vid_name)

    venv = model._wrap_env(venv)
    
    steps = 0
    obs = venv.reset() # Starts video recorder

    while steps <= config.total_steps:
        obs_tensor = obs_as_tensor(np.repeat(obs, original_processes, axis=0), model.device)
        model._last_policy_mems = model._last_policy_mems.to(model.device)
        actions, ext_values, int_values, log_probs, policy_mems = model.policy.forward( obs_tensor, model._last_policy_mems, deterministic=config.deterministic )
        actions = actions.cpu().numpy()
        if isinstance(venv.action_space, gym.spaces.Box):
            actions = np.clip(actions, venv.action_space.low, venv.action_space.high)
        actions = actions[0:1]
        new_obs, rewards, costs, dones, infos = venv.step(actions)
        

        intrinsic_rewards, model_mems = \
            model.create_intrinsic_rewards(new_obs, actions, dones)

        if policy_mems is not None:
            model._last_policy_mems = policy_mems.detach().clone()
        if model_mems is not None:
            model._last_model_mems = model_mems.detach().clone()
        
        for i, d in enumerate(dones):
            if d:
                venv.send_reset(env_id=i)
                new_obs[i] = venv.recv_obs(env_id=i)

        obs = new_obs
        steps += 1
    
    venv.close()


@click.command()
# Both directories should have the same structure
@click.option('--model', type=str, help='Path to the zip file that contains the model snapshot')
@click.option('--seed', default=0, type=int, help='Environment initial seed')
# Process options\
@click.option('--total_steps', default=5000, type=int, help='Total number of frames to run for testing')
@click.option('--deterministic', default=1, type=int, help="Whether to use the most likely action instead of PPO random action sampling. 0 = Stochastic/Random, 1 = Deterministic")

def main(
    model, seed, total_steps, deterministic,
):
    args = locals().items()
    config = TrainingConfig()
    for k, v in args: setattr(config, k, v)

    make_video(config)

if __name__ == '__main__':
    main()