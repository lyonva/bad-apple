from gymnasium.core import Wrapper
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.core.actions import Actions

# Adds a secondary reward, representing the cost of the agent
# Constraint_list parameter determines which elements of the environment are deemed unsafe
class MiniGridSafetyCostWrapper(Wrapper):
    def __init__(self, env, enable_constraints : int = 0, collision_cost : float = 0, termination_cost : float = 0):
        super().__init__(env)
        self.enable_constraints = enable_constraints
        self.collision_cost = collision_cost
        self.termination_cost = termination_cost
    
    def reset(self, *, seed = None, options = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        cost = 0

        if self.enable_constraints:

            if self.termination_cost > 0:
                if terminated and (reward <= 0):
                    cost += self.termination_cost
            
            if self.collision_cost > 0:
                if self._last_obs[3][5][0] in [OBJECT_TO_IDX["wall"], OBJECT_TO_IDX["door"], OBJECT_TO_IDX["key"], OBJECT_TO_IDX["box"], OBJECT_TO_IDX["ball"]] \
                        and action == Actions.forward:
                    cost += self.collision_cost
                
        
        self._last_obs = obs

        return obs, reward, cost, terminated, truncated, info
