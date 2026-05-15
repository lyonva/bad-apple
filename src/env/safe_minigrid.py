"""
This file is used to define and register custom MiniGrid environments.
The code was created based on the official implementation of gym-minigrid
(version 1.0.3) and may differ from the latest version.
"""

import math
from abc import abstractmethod
from typing import Any, Any, Iterable, SupportsFloat, TypeVar, ClassVar

import gymnasium as gym
import numpy as np

from gymnasium import spaces
from gymnasium.core import ActType, ObsType

from minigrid.minigrid_env import MiniGridEnv
from minigrid.envs import DoorKeyEnv
from minigrid.core.world_object import Point, WorldObj, Goal, Door, Key, Ball, Box, Lava, Floor
from minigrid.core.grid import Grid, Wall
from minigrid.core.actions import Actions
from minigrid.core.constants import COLORS, COLOR_NAMES, DIR_TO_VEC, TILE_PIXELS
from minigrid.core.roomgrid import RoomGrid
from minigrid.core.mission import MissionSpace
from minigrid.utils.rendering import fill_coords, point_in_line, point_in_rect, point_in_circle
from minigrid import register
from gymnasium.core import ActType, ObsType
from minigrid.core.actions import Actions

def _pos_isolated(self, pos):
    """Returns False if position is surrounded by any object other than walls"""
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            new_pos = (pos[0]+dx, pos[1]+dy)
            tile = self.grid.get(*new_pos)
            if tile is None:
                continue
            if tile.type != "wall":
                return True

    return False

class Water(WorldObj):
    def __init__(self):
        super().__init__("floor", "blue")

    def can_overlap(self):
        return True

    def render(self, img):
        c = (0, 128, 255)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))

class Poison(WorldObj):
    def __init__(self):
        super().__init__("floor", "purple")

    def can_overlap(self):
        return True

    def render(self, img):
        c = (128, 0, 128)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))

class Bog(WorldObj):
    def __init__(self):
        super().__init__("floor", "yellow")

    def can_overlap(self):
        return True

    def render(self, img):
        c = (128, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(2):
            ylo = 0.3 + 0.4 * i
            yhi = 0.4 + 0.4 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class Ice(WorldObj):
    def __init__(self):
        super().__init__("floor", "teal")

    def can_overlap(self):
        return True

    def render(self, img):
        c = (128, 128, 255)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        # Shine effect
        fill_coords(img, point_in_line(0.05, 0.15, 0.15, 0.05, r=0.02), (255, 255, 255))
        fill_coords(img, point_in_line(0.1, 0.2, 0.2, 0.1, r=0.02), (255, 255, 255))
        fill_coords(img, point_in_line(0.9, 0.8, 0.8, 0.9, r=0.02), (255, 255, 255))
        fill_coords(img, point_in_line(0.95, 0.85, 0.85, 0.95, r=0.02), (255, 255, 255))
        

class Button(WorldObj):
    def __init__(self, color="red"):
        super().__init__("ball", color)
    
    def can_overlap(self):
        return False
    
    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS["grey"])
        fill_coords(img, point_in_circle(0.5, 0.5, 0.25), COLORS[self.color])

class SafeChoiceEnv(MiniGridEnv):
    def __init__(self,
        max_steps : int | None = None,
        **kwargs,
    ):
        width = 5
        height = 4
        self.goal1_pos = (3, 1) # Moat
        self.goal2_pos = (2, 2) # Neutral
        self.goal3_pos = (1, 1) # Guard

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 10
        
        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
    
    @staticmethod
    def _gen_mission():
        return "get to the green goal square"
    
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(color="blue"), *self.goal1_pos)
        self.put_obj(Goal(), *self.goal2_pos)
        self.put_obj(Goal(color="red"), *self.goal3_pos)

        # Place obstacles
        for x, y in ((1, 2), (3, 2), (4, 2)):
            self.put_obj(Wall(), x, y)

        # Place the agent
        self.agent_pos = (2, 1)
        self.agent_dir = 1

        self.mission = "get to the green goal square"
    
    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """
        if self.agent_pos == self.goal1_pos:
            return 1
        if self.agent_pos == self.goal2_pos:
            return 0.1
        if self.agent_pos == self.goal3_pos:
            return 0.2
        return 0
    
    def _cost(self):
        cost = 0
        if self.agent_pos == self.goal3_pos:
            cost = 4
        
        if self.agent_pos == self.goal1_pos:
            cost = 10

        return cost

register(
    id='MiniGrid-SafeChoice-v0',
    entry_point='src.env.safe_minigrid:SafeChoiceEnv',
)

class SafeBogMazeEnv(MiniGridEnv):

    def __init__(self,
        max_steps : int | None = None,
        **kwargs,
    ):
        width = 5
        height = 7
        self.goal_pos = (1, 1)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 100
        
        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
    
    @staticmethod
    def _gen_mission():
        return "get to the green goal square"
    
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the tp[-right corner
        self.put_obj(Goal(), *self.goal_pos)

        # Place the agent
        self.agent_pos = (2, 5)
        self.agent_dir = 3

        # Place the water tiles
        for y in range(2, 6):
            self.put_obj(Water(), 1, y)

        # Place walls
        walls = [(2,2), (2,4)]
        
        for x, y in walls:
            self.put_obj(Wall(), x, y)
        
        # Put bog tile
        self.put_obj(Bog(), 3, 2)

        self.mission = "get to the green goal square"
    
    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """

        return 1 - 0.9 * (self.step_count / self.max_steps)

    def _cost(self):
        # Check water
        cost = 0
        new_cell = self.grid.get(*self.agent_pos)
        if new_cell is not None and new_cell.type == "floor" and new_cell.color == "blue":
            cost = 1
        return cost
    
    
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        obs, reward, terminated, truncated, info = super().step(action)

        # Bog penalty
        if self.agent_pos == (3,2):
            self.step_count += 3

        return obs, reward, terminated, truncated, info

register(
    id='MiniGrid-SafeBogMaze-v0',
    entry_point='src.env.safe_minigrid:SafeBogMazeEnv',
)

class SafeWaterMaze5x7Env(MiniGridEnv):

    def __init__(self,
        max_steps : int | None = None,
        **kwargs,
    ):
        width = 5
        height = 7
        self.goal_pos = (1, 2)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 100
        
        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
    
    @staticmethod
    def _gen_mission():
        return "get to the green goal square"
    
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the tp[-right corner
        self.put_obj(Goal(), *self.goal_pos)

        # Place the agent
        self.agent_pos = (2, 5)
        self.agent_dir = 3

        # Place the water tiles
        for y in range(3, 6):
            self.put_obj(Water(), 1, y)

        # Place walls
        walls = [(2,2), (2,4)]
        
        for x, y in walls:
            self.put_obj(Wall(), x, y)

        self.mission = "get to the green goal square"

    def _cost(self):
        # Check water
        cost = 0
        new_cell = self.grid.get(*self.agent_pos)
        if new_cell is not None and new_cell.type == "floor" and new_cell.color == "blue":
            cost = 1
        return cost

register(
    id='MiniGrid-SafeWaterMaze-5x7-v0',
    entry_point='src.env.safe_minigrid:SafeWaterMaze5x7Env',
)

class SafeMoatMaze8x8Env(MiniGridEnv):
    '''
    The Safe Moat Maze: A maze with three viable shortest paths. The short but high cost moat, the medium cost guard, and the long but costless path.
    Used for safe RL and risk.

    Based on the original Guardmaze code:
    https://github.com/ido90/CeSoR/blob/main/Examples/GuardedMaze/GuardedMaze.py
    '''

    def __init__(self,
        guard_prob=0.25,
        max_steps : int | None = None,
        **kwargs,
    ):
        width = 8
        height = 8
        self.guard_pos = (5, 6)
        self.moat_pos = (5, 4)
        self.goal_pos = (6, 3)
        self.guard_prob = guard_prob

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 100
        
        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            agent_view_size=9,
            **kwargs,
        )
    
    @staticmethod
    def _gen_mission():
        return "get to the green goal square"
    
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), *self.goal_pos)

        # Place obstacles
        for x, y in ((3, 2), (4, 2), (5, 2), (5, 3), (5, 5), (5, 6)):
            self.put_obj(Wall(), x, y)
        
        # Color the guard and moat
        self.put_obj(Floor("red"), *self.guard_pos)
        self.put_obj(Water(), *self.moat_pos)

        # Place the agent
        # self.agent_pos = (2, 5)
        self.place_agent((1,3), (4,4))
        # self.agent_dir = 0

        self.mission = "get to the green goal square"
    
    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """

        # return 1.12 - 0.03 * min(32, self.step_count)
        return 1 - 0.9 * (self.step_count / self.max_steps)
    
    def _cost(self):
        cost = 0
        if self.agent_pos == self.guard_pos:
            if self.guard_prob == 0:
               cost = 4
            else:
                if self._rand_float(0, 1) < self.guard_prob:
                    cost = 10
                else:
                    cost = 2
        
        if self.agent_pos == self.moat_pos:
            cost = 10

        return cost

class SafeDetMoatMaze8x8Env(SafeMoatMaze8x8Env):
    def __init__(self, max_steps : int | None = None, **kwargs,):
        super().__init__(guard_prob=0, max_steps=max_steps, **kwargs)

register(
    id='MiniGrid-SafeMoatMaze-8x8-v0',
    entry_point='src.env.safe_minigrid:SafeMoatMaze8x8Env',
)
register(
    id='MiniGrid-SafeDetMoatMaze-8x8-v0',
    entry_point='src.env.safe_minigrid:SafeDetMoatMaze8x8Env',
)

class SafeMockjocoGoalEnv(MiniGridEnv):
    def __init__(self,
        width : int,
        height : int,
        num_poison : int | None = None,
        timeout: int | None = None,
        max_steps : int | None = None,
        **kwargs,
    ):

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 2 * width * height
        if num_poison is None:
            num_poison = (width + height) // 2
        if timeout is None:
            timeout = 2*(width + height)
        
        self.num_poison = num_poison
        self.timeout = timeout
        
        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

        self.goal_pos = None
        self.poison_pos = []
    
    @staticmethod
    def _gen_mission():
        return "get to the green goal square while avoiding poison"
    
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent
        self.agent_pos = self.place_agent(top=(1,1), size=(self.width-2, self.height-2))

        self._shuffle_layout()

        self.mission = "get to the green goal square while avoiding poison"
        
    
    def _shuffle_layout(self):
        # Remove everything
        self.timer = 0

        if self.goal_pos is not None:
            self.grid.set(self.goal_pos[0], self.goal_pos[1], None)

        
        while len(self.poison_pos) > 0:
            pos = self.poison_pos.pop()
            self.grid.set(pos[0], pos[1], None)

        for _ in range(self.num_poison):
            self.poison_pos.append(self.place_obj(Poison(), (1,1), (self.width-2, self.height-2)))

        self.goal_pos = self.place_obj(Goal(), (1,1), (self.width-2, self.height-2))
    
    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """
        return 0
    
    def _cost(self):
        cost = 0
        if self.agent_pos in self.poison_pos:
            cost = 1

        return cost
    
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        
        obs, reward, terminated, truncated, info = super().step(action)
        curr_pos = self.agent_pos

        if terminated: # Goal found
            reward = 1
            terminated = False
            self._shuffle_layout()
            obs = self.gen_obs()
        else:
            # Check timer
            # In case of lock-out generations
            self.timer += 1
            if self.timer >= self.timeout:
                self._shuffle_layout()
                obs = self.gen_obs()

        return obs, reward, terminated, truncated, info

class SafeMockjocoGoal9x9Env(SafeMockjocoGoalEnv):
    def __init__(self, **kwargs,):
        super().__init__(width=9, height=9, agent_view_size=7, **kwargs)

register(
    id='MiniGrid-SafeMockjocoGoal-9x9-v0',
    entry_point='src.env.safe_minigrid:SafeMockjocoGoal9x9Env',
)


class SafeMockjocoButton(MiniGridEnv):

    def __init__(self,
        width : int,
        height : int,
        num_buttons : int,
        num_poison : int | None = None,
        max_steps : int | None = None,
        **kwargs,
    ):
        mission_space = MissionSpace(mission_func=self._gen_mission)

        if num_poison is None:
            num_poison = (width + height) // 2 - num_buttons
        if max_steps is None:
            max_steps = 2 * width * height
        
        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        self.num_buttons = num_buttons
        self.num_poison = num_poison

        self.button_pos = []
        self.poison_pos = []
        self.goal_idx = -1
    
    @staticmethod
    def _gen_mission():
        return "get to the green button while avoiding poison"
    
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.button_pos = []
        self.poison_pos = []

        # Place button tiles
        for _ in range(self.num_buttons):
            pos = self.place_obj(Button(color="red"), (1,1), (self.width-2, self.height-2), reject_fn=_pos_isolated)
            self.button_pos.append(pos)

        # Place the agent
        self.agent_pos = self.place_agent(top=(1,1), size=(self.width-2, self.height-2))

        self._shuffle_layout()

        self.mission = "get to the green button while avoiding poison"
    
    def _shuffle_layout(self):
        # Remove goal, set button
        if self.goal_idx > -1:
            self.put_obj(Button(color="red"), self.button_pos[self.goal_idx][0], self.button_pos[self.goal_idx][1])

        # Remove poison
        while len(self.poison_pos) > 0:
            pos = self.poison_pos.pop()
            self.grid.set(pos[0], pos[1], None)

        # Set one of the buttons as goal
        new_goal = self._rand_int(0, self.num_buttons-1)
        if new_goal >= self.goal_idx: new_goal += 1
        self.goal_idx = new_goal
        self.put_obj(Button(color="green"), self.button_pos[self.goal_idx][0], self.button_pos[self.goal_idx][1])

        for _ in range(self.num_poison):
            self.poison_pos.append(self.place_obj(Poison(), (1,1), (self.width-2, self.height-2), reject_fn=_pos_isolated))

    
    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """
        return 0
    
    def _cost(self):
        cost = 0
        if self.agent_pos in self.poison_pos:
            cost = 1
        if self.last_act == Actions.toggle:
            if self.last_reward < 1:
                cost = 1

        return cost
    
    def reset(self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> spaces.Box:
        self.last_act = Actions.done
        self.last_reward = 0
        return super().reset(seed=seed)
    
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.last_act = action
        obs, reward, terminated, truncated, info = super().step(action)
        curr_pos = self.agent_pos
        front_pos = tuple(self.front_pos)

        if action == Actions.toggle and front_pos == self.button_pos[self.goal_idx]: # Goal found
            reward = 1
            self._shuffle_layout()
            obs = self.gen_obs()
        self.last_reward = reward

        return obs, reward, terminated, truncated, info

class SafeMockjocoButton9x9Env(SafeMockjocoButton):
    def __init__(self, **kwargs,):
        super().__init__(width=9, height=9, num_buttons=3, agent_view_size=7, **kwargs)

class SafeMockjocoButton11x11Env(SafeMockjocoButton):
    def __init__(self, **kwargs,):
        super().__init__(width=11, height=11, num_buttons=4, agent_view_size=9, **kwargs)

register(
    id='MiniGrid-SafeMockjocoButton-9x9-v0',
    entry_point='src.env.safe_minigrid:SafeMockjocoButton9x9Env',
)
register(
    id='MiniGrid-SafeMockjocoButton-11x11-v0',
    entry_point='src.env.safe_minigrid:SafeMockjocoButton11x11Env',
)

class SafeCliffWalkEnv(MiniGridEnv):
    def __init__(self,
        width,
        height,
        slip_proba,
        max_steps : int | None = None,
        **kwargs,
    ):
        self.goal_pos = (width - 2, height - 2)
        self.slip_proba = slip_proba

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 2 * width * height
        
        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
    
    @staticmethod
    def _gen_mission():
        return "get to the green goal square, avoid poison tiles"
    
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), *self.goal_pos)

        # Place the poison rows
        for i in range(2, self.width - 2):
            self.grid.set(i, self.height-2, Poison())

        # Place the agent
        self.agent_pos = (1, self.height-2)
        self.agent_dir = 0

        self.mission = "get to the green goal square, avoid poison tiles"
    
    def _cost(self):
        # Check poison
        cost = 0
        new_cell = self.grid.get(*self.agent_pos)
        if new_cell is not None and new_cell.type == "floor" and new_cell.color == "purple":
            cost = 1
        return cost
    
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        curr_pos = self.agent_pos
        prev_dir = self.agent_dir
        slip = False
        # Slip if moving into/within the cliff
        # Note this happens regardless of action, and instead of the action
        # This makes staying near the bottom risky
        if (2 < curr_pos[0] < self.width - 2) and curr_pos[1] < self.height - 2:
            slip = self._rand_float(0,1) < self.slip_proba
            if slip:
                # Agent slips down 1 tile
                self.agent_dir = 1 # Down
                action = Actions.forward

        obs, reward, terminated, truncated, info = super().step(action)

        # Restore direction
        if slip:
            self.agent_dir = prev_dir
        return self.gen_obs(), reward, terminated, truncated, info

class SafeCliffWalk12x9S25Env(SafeCliffWalkEnv):
    def __init__(self, **kwargs,):
        super().__init__(width=12, height=9, slip_proba=0.25, agent_view_size=9, **kwargs)
register(
    id='MiniGrid-SafeCliffWalk-12x9-S25-v0',
    entry_point='src.env.safe_minigrid:SafeCliffWalk12x9S25Env',
)
