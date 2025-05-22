from src.algo.reward_shaping.base_shaping import IntrinsicRewardShaping
import numpy as np

class ADOPES(IntrinsicRewardShaping):
    def __init__(self, gamma:float, n_envs:int, adopes_coef_inc:int=1, adops_epsilon=0.000001):
        self.gamma = gamma
        self.n_envs = n_envs
        self.epsilon = adops_epsilon
        self.scale_coef = 1 if adopes_coef_inc>=1 else 0
        self.adopes_coef_inc = adopes_coef_inc
    
    def shape_rewards(self, rewards, intrinsic_rewards, ext_values, int_values, next_ext_values, next_int_values, dones):
        adopes_rewards = intrinsic_rewards.copy()
        q_e = rewards + next_ext_values.cpu().numpy() * (1-dones)
        q_i = next_int_values.cpu().numpy() * (1-dones) # Actually q_i,t+1
        v_e = ext_values.cpu().numpy()
        v_i = int_values.cpu().numpy()

        partial_f2 = v_e - q_e + v_i - self.gamma*q_i - intrinsic_rewards
        f2 = np.zeros(self.n_envs)
        for env in range(self.n_envs):
            if q_e[env] < v_e[env]:
                f2[env] = min(0, partial_f2[env] - self.epsilon)
            else:
                f2[env] = max(0, partial_f2[env])
        adopes_rewards += self.scale_coef * f2
        return adopes_rewards

    def on_rollout_end(self):
        self.scale_coef = min(1, self.scale_coef + self.adopes_coef_inc)
