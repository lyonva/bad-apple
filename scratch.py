import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
# from src.env.safety_constraints import MiniGridSafetyCostWrapper
from minigrid.core.actions import Actions

# # env = gym.make("MiniGrid-Empty-16x16-v0", max_episode_steps=5)
# env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
# env = ImgObsWrapper(env)
# env = MiniGridSafetyCostWrapper(env, ['wall_colission', 'unsuccessful_termination'])

# obs, _ = env.reset()

# for _ in range(20):
#     obs, r, c, ter, tur, _ = env.step(Actions.forward)

#     print(c, ter, tur)

# obs, r, c, _, _, _ = env.step(Actions.right)

# print(c)

# for _ in range(20):
#     obs, r, c, ter, tur, _ = env.step(Actions.forward)

#     print(c, ter, tur)


# import pickle

# with open('params_orig.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)
# loaded_dict['collision_cost'] = 0.75
# with open('params.pkl', 'wb') as f:
#     pickle.dump(loaded_dict, f)

from src.algo.ppo_trainer import PPOTrainer
from src.utils.enum_types import ModelType, ShapeType

model_main = PPOTrainer.load( "snapshot-1250.zip", env=None )
im = ModelType.get_str_name(model_main.int_rew_source)
rs = ShapeType.get_str_name(model_main.int_shape_source)
print(im, rs)
