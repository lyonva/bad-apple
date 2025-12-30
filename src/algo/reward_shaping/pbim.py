from src.algo.reward_shaping.base_shaping import IntrinsicRewardShaping
import numpy as np

class PBIM(IntrinsicRewardShaping):
    def __init__(self, gamma:float, n_envs:int):
        self.gamma = gamma
        self.n_envs = n_envs
        self.pbim_buffer = [[] for _ in range(n_envs)]
    
    def shape_rewards(self, int_rewards, curr_dones):
        discounted_rewards = int_rewards.copy()

        # Check for episode ends and discount all others if so
        for env in range(self.n_envs):
            if curr_dones[env] != 0:
                self.pbim_buffer[env].append(int_rewards[env])
            else:
                # Discount all and reset
                new_int_reward = 0
                ep_lenght = len(self.pbim_buffer[env]) + 2
                for n in range(len(self.pbim_buffer[env])): # Discount all intrinsic rewards
                    new_int_reward -= (self.gamma**(n+1-ep_lenght))*self.pbim_buffer[env][n]
                discounted_rewards[env] = new_int_reward
                self.pbim_buffer[env] = []


        return discounted_rewards

