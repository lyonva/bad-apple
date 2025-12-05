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

from src.algo.ppo_model import PPOModel
from src.algo.ppo_trainer import PPOTrainer
from src.algo.ppo_rollout import PPORollout
from src.utils.configs import TrainingConfig
from stable_baselines3.common.utils import set_random_seed, obs_as_tensor
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape

from src.algo.oracle_model import Oracle

warnings.filterwarnings("ignore", category=DeprecationWarning)

# From: https://stackoverflow.com/questions/47631914/how-to-pass-several-list-of-arguments-to-click-option
class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)
        

def test_vs_oracle(config):
    # Get all methods
    models_dir = os.path.join(config.models_dir, f"Minigrid-{config.game_name}-v0")
    techs = set([x.split("-")[0] for x in os.listdir(models_dir)])
    seeds = set([int(x.split("-")[1]) for x in os.listdir(models_dir)])
    data = []

    env = gym.make(config.env_name)
    env = ImgObsWrapper(env)
    oracle_env = gym.make(config.env_name)
    oracle_env = FullyObsWrapper(oracle_env)
    oracle_env = ImgObsWrapper(oracle_env)
    oracle = Oracle(env, config.env_name)

    obs = env.reset()

    env.can_see_walls = True
    env.image_noise_scale = 0

    np.random.seed(1337)

    obs_shape = get_obs_shape(env.observation_space)
    action_dim = get_action_dim(env.action_space)

    all_oracle_steps = []
    all_oracle_rewards = []
    for ep in range(1, config.total_episodes+1):
        # Oracle
        done = False
        oracle.reset()
        obs, _ = oracle_env.reset(seed=ep)
        oracle_steps = 0
        oracle_total_reward = 0
        # print(oracle_env.unwrapped.pprint_grid())
        while not(done):
            action = oracle.forward(obs)
            obs, reward, terminated, truncated, _ = oracle_env.step(action)
            done = terminated or truncated
            oracle_steps += 1
            oracle_total_reward += reward
        all_oracle_steps.append(oracle_steps)
        all_oracle_rewards.append(oracle_total_reward)

    # Data collection
    # Per model
    for tech in techs:

        for snap in config.snaps:

            for seed in seeds:
                model = PPOTrainer.load( os.path.join(models_dir, f"{tech}-{seed}", f"snapshot-{snap}"), env=None )
                model.policy.eval()

                for ep in range(1, config.total_episodes+1):
                    # Agent
                    done = False
                    obs, _ = env.reset(seed=ep)
                    steps = 0
                    total_reward = 0
                    while not(done):
                        obs_tensor = obs_as_tensor(np.array([obs for _ in range(16)]), model.device)
                        obs_tensor=obs_tensor.permute(0,3,1,2)
                        model._last_policy_mems = model._last_policy_mems.to(model.device)

                        actions, ext_values, int_values, log_probs, policy_mems = model.policy.forward( obs_tensor, model._last_policy_mems, deterministic=config.deterministic )
                        actions = actions.cpu().numpy()
                        if isinstance(env.action_space, gym.spaces.Box):
                            actions = np.clip(actions, env.action_space.low, env.action_space.high)
                        action = actions[0].item()
                        obs, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        if policy_mems is not None:
                            model._last_policy_mems = policy_mems.detach().clone()
                        steps += 1
                        total_reward += reward
                
                    data.append([tech, seed, snap, ep, all_oracle_steps[ep-1], all_oracle_rewards[ep-1], steps, total_reward])
    
    pos_df = pd.DataFrame.from_records(data, columns=["im", "seed", "snapshot", "episode", "steps_oracle", "reward_oracle", "steps", "reward"])
    pos_df.to_csv(f"analysis/rewards-{config.game_name}.csv")



@click.command()
# Testing params
@click.option('--game_name', default="DoorKey-8x8", type=str, help='e.g. DoorKey-8x8, FourRooms, RedBlueDoors-8x8')
# Both directories should have the same structure
@click.option('--models_dir', type=str, default="analysis\logs", help='Path to the folder that contains the models')
@click.option('--snaps', default='[]', cls=PythonLiteralOption, help='List of model checkpoints')
# Process options
@click.option('--total_episodes', default=int(100), type=int, help='Total number of episodes to run for testing, per model and seed')
@click.option('--deterministic', default=1, type=int, help="Whether to use the most likely action instead of PPO random action sampling. 0 = Stochastic/Random, 1 = Deterministic")

def main(
    game_name, models_dir, snaps, total_episodes, deterministic,
):
    args = locals().items()
    config = TrainingConfig()
    for k, v in args: setattr(config, k, v)
    config.env_source = "minigrid"
    config.deterministic = bool(config.deterministic)

    config.init_env_name(game_name, None)

    test_vs_oracle(config)

#  python .\test_vs_oracle.py --game_name=DoorKey-8x8 --snaps=[500,1000,2500,5000,10000]
if __name__ == '__main__':
    main()