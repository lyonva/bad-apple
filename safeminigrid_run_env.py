from src.env.manual_control import SafeManualControl
from minigrid.minigrid_env import MiniGridEnv
from src.env.safe_minigrid import *
import gymnasium as gym
import minigrid

env = gym.make("MiniGrid-SafeBogMaze-v0",render_mode="human")

# Manual seed
# env.reset(seed=23)

# enable manual control for testing
manual_control = SafeManualControl(env)
manual_control.start()