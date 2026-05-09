from src.env.manual_control import SafeManualControl
from minigrid.minigrid_env import MiniGridEnv
from src.env.safe_minigrid import *
from src.env.wrapper import MiniGridCostWorkaroundWrapper
import gymnasium as gym
import minigrid

# env = gym.make("MiniGrid-SafeBogMaze-v0",render_mode="human")
# env = gym.make("MiniGrid-SafeChoice-v0",render_mode="human")
env = gym.make("MiniGrid-SafeWaterMaze-5x7-v0",render_mode="human")

env = MiniGridCostWorkaroundWrapper(env)
# Manual seed
# env.reset(seed=23)

# enable manual control for testing
manual_control = SafeManualControl(env)
manual_control.start()