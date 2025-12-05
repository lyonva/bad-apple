from src.algo.reward_shaping.base_shaping import IntrinsicRewardShaping
import numpy as np

class GRM(IntrinsicRewardShaping):
    def __init__(self, gamma:float, n_envs:int, grm_delay:int=1):
        self.gamma = gamma
        self.n_envs = n_envs
        self.grm_delay = grm_delay
        self.grm_buffer = np.zeros((n_envs, grm_delay))
    
    def shape_rewards(self, int_rewards, curr_dones):
        # curr_dones = curr_dones.cpu().numpy()
        undiscounted_rewards = int_rewards.copy()
        grm_rewards = int_rewards - self.grm_buffer[:,-1]/(self.gamma**(self.grm_delay))
        # Check for episode ends and discount all others if so
        for env in range(self.n_envs):
            if curr_dones[env] != 0:
                # Discount all and reset
                grm_rewards[env] -= np.sum( self.grm_buffer[env,0:-1] / (self.gamma** np.arange(1,self.grm_delay)) )
                grm_rewards[env] -= undiscounted_rewards[env] # Also remove the just-obtained reward
                self.grm_buffer[env,:] = np.zeros(self.grm_delay) # Clear buffer
                
                

        # Log rewards on buffer, rewards on timestep D are removed
        self.grm_buffer = np.roll(self.grm_buffer, 1, axis=1)
        self.grm_buffer[:,0] = undiscounted_rewards * (1-curr_dones)

        return grm_rewards

