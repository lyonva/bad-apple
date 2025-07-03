from src.algo.reward_shaping.base_shaping import IntrinsicRewardShaping
import numpy as np

class PIES(IntrinsicRewardShaping):
    def __init__(self, gamma:float, n_envs:int, pies_decay:int=2000):
        self.gamma = gamma
        self.n_envs = n_envs
        self.scale_coef = 1
        self.pies_decay = pies_decay
    
    def shape_rewards(self, intrinsic_rewards):
        return intrinsic_rewards * self.scale_coef

    def on_rollout_end(self):
        self.scale_coef = max(0, self.scale_coef - 1.0/self.pies_decay)
