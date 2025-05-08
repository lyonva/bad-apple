from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
import gymnasium as gym
import minigrid

# env = gym.make("MiniGrid-DoorKey-6x6-v0",render_mode="human")
env = gym.make("MiniGrid-FourRooms-v0",render_mode="human")
# env = gym.make("MiniGrid-RedBlueDoors-8x8-v0",render_mode="human")

# enable manual control for testing
manual_control = ManualControl(env)
manual_control.start()