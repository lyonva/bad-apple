import click
import ast
import warnings
import os
import numpy as np
import torch as th
import gymnasium as gym

from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper, ReseedWrapper
from stable_baselines3.common.env_util import make_vec_env
from src.env.subproc_vec_env import CustomSubprocVecEnv
from src.env.minigrid_envs import *

from src.algo.ppo_model import PPOModel
from src.algo.ppo_trainer import PPOTrainer
from src.algo.ppo_rollout import PPORollout
from src.utils.configs import TrainingConfig
from stable_baselines3.common.utils import set_random_seed, obs_as_tensor

warnings.filterwarnings("ignore", category=DeprecationWarning)

# From: https://stackoverflow.com/questions/47631914/how-to-pass-several-list-of-arguments-to-click-option
class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)

class TestingRecord:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.ext_rewards = []
        self.int_rewards = []
        self.dones = []

        self.positions = []


        

def float_zeros(tensor_shape, config):
    return th.zeros(tensor_shape, device=config.device, dtype=th.float32)

def test(config):

    data_main, data_base = {}, {}

    for snap in config.snaps:

        data_main[snap], data_base[snap] = {}, {}

        for seed in config.seeds:

            record_main, record_baseline = TestingRecord(), TestingRecord()

            env = make_vec_env(config.env_name,
                               wrapper_class=lambda x: ImgObsWrapper(x),
                               vec_env_cls=CustomSubprocVecEnv,
                               n_envs=config.num_processes,
                            )
            model_baseline = PPOTrainer.load( os.path.join(config.models_dir, config.baseline, f"{seed}", f"snapshot-{snap}"), env=None )
            model_main = PPOTrainer.load( os.path.join(config.models_dir, config.method, f"{seed}", f"snapshot-{snap}"), env=None )

            env = model_baseline._wrap_env(env)
            obs = env.reset()

            env.can_see_walls = model_baseline.can_see_walls
            env.image_noise_scale = model_baseline.image_noise_scale
            env.image_rng = np.random.default_rng(seed=seed + 1313)
            np.random.seed(seed)
            seeds = np.random.rand(config.num_processes)
            seeds = [int(s * 0x7fffffff) for s in seeds]
            np.random.seed(seed)
            env.set_seeds(seeds)
            env.waiting = True
            for i in range(config.num_processes):
                env.send_reset(env_id=i)
            for i in range(config.num_processes):
                obs[i] = env.recv_obs(env_id=i)
            env.waiting = False

            # Data collection
            # Per model
            for model in [model_main, model_baseline]:
                obs = env.reset()
                steps = 0
                model.policy.eval()

                while steps <= config.total_steps:
                    obs = obs_as_tensor(obs, config.device)
                    actions, values, log_probs, policy_mems = model.policy.forward( obs, model._last_policy_mems )
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

                    obs = new_obs
                    steps += config.num_processes



@click.command()
# Testing params
@click.option('--game_name', default="DoorKey-8x8", type=str, help='e.g. DoorKey-8x8, FourRooms, RedBlueDoors-8x8')
# Both directories should have the same structure
@click.option('--models_dir', type=str, help='Path to the folder that contains the models')
@click.option('--baseline', type=str, help='Name of the baseline method')
@click.option('--method', type=str, help='Name of the main method')
@click.option('--seeds', default='[]', cls=PythonLiteralOption, help='List of seeds')
@click.option('--snaps', default='[]', cls=PythonLiteralOption, help='List of model snapshots')
# Process options
@click.option('--num_processes', default=16, type=int, help='Number of testing processes (workers)')
@click.option('--total_steps', default=int(5000), type=int, help='Total number of frames to run for testing')

def main(
    game_name, models_dir, baseline, method, seeds, snaps, num_processes, total_steps,
):
    args = locals().items()
    config = TrainingConfig()
    for k, v in args: setattr(config, k, v)
    config.env_source = "minigrid"

    config.init_env_name(game_name, None)

    test(config)

# python .\test.py --game_name=DoorKey-16x16 --models_dir=analysis\models\DoorKey-16x16 --baseline=NoModel --method=GRM --seeds=[1] --snaps=[3,15,305]
if __name__ == '__main__':
    main()