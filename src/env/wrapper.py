from gymnasium.core import ActType, ObsType, Wrapper
from typing import SupportsFloat, Any

class MiniGridCostWorkaroundWrapper(Wrapper):
    """Hacks to return cost in addition to reward"""

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        cost = self.env.unwrapped._cost()

        return obs, reward, cost, terminated, truncated, info
