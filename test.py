import click
import ast
import warnings
import os

from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper, ReseedWrapper
from stable_baselines3.common.env_util import make_vec_env
from src.env.subproc_vec_env import CustomSubprocVecEnv
from src.env.minigrid_envs import *

from src.algo.ppo_model import PPOModel
from src.algo.ppo_trainer import PPOTrainer
from src.algo.ppo_rollout import PPORollout
from src.utils.configs import TrainingConfig
from stable_baselines3.common.utils import set_random_seed

warnings.filterwarnings("ignore", category=DeprecationWarning)

# From: https://stackoverflow.com/questions/47631914/how-to-pass-several-list-of-arguments-to-click-option
class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)

def test(config):

    for snap in config.snaps:

        for seed in config.seeds:

            env = make_vec_env(config.env_name,
                               wrapper_class=lambda x: ImgObsWrapper(ReseedWrapper(x, seeds=[seed])),
                               vec_env_cls=CustomSubprocVecEnv,
                               n_envs=config.num_processes,
                            )
            model_baseline = PPOTrainer.load( os.path.join(config.models_dir, config.baseline, f"{seed}", f"snapshot-{snap}.zip"), env=None )
            model_main = PPOTrainer.load( os.path.join(config.models_dir, config.method, f"{seed}", f"snapshot-{snap}.zip"), env=None )

            obs = env.reset()
            steps = 0


            # Main model data collection
            while steps <= config.total_steps:
                
                actions = model_main.predict(obs)
                new_obs, rewards, dones, infos = env.step()

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


if __name__ == '__main__':
    main()