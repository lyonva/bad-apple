from enum import Enum
from minigrid.core.constants import OBJECT_TO_IDX, DIR_TO_VEC, COLOR_TO_IDX, STATE_TO_IDX
from minigrid.core.actions import Actions
from queue import PriorityQueue
from heapq import heapify

import numpy as np

from dataclasses import dataclass, field
from typing import Any

class OraclePlanSteps(Enum):
    NOOP = 0
    PATHFIND = 1
    PICKUP = 2
    DROP = 3
    OPEN = 4
    CLOSE = 5

def forward_pos(x, y, r):
    vec_dir = DIR_TO_VEC[r]
    return (x + vec_dir[0], y + vec_dir[1], r)

def backward_pos(x, y, r):
    vec_dir = DIR_TO_VEC[r]
    return (x - vec_dir[0], y - vec_dir[1], r)

class Oracle:

    def __init__(self, env, env_name):
        self.env = env
        env_name = env_name.split("-")
        if len(env_name) == 3:
            env_family, env_name, version = env_name
        if len(env_name) == 4:
            env_family, env_name, map_size, version = env_name
        if len(env_name) == 1:
            env_name = env_name[0]
        self.env_name = env_name
        self.plan = None
        self.path = []
    
    def _plan(self, obs):
        # Starting agent position
        sx, sy, sr = -1, -1, -1
        done = False
        for i in range(len(obs)):
            for j in range(len(obs[i])):
                if obs[i][j][0] == OBJECT_TO_IDX["agent"]:
                    sx, sy = i, j
                    sr = obs[i][j][2]
                    done = True
                    break
            if done: break

        if self.env_name == "Empty" or self.env_name == "FourRooms":
            # Find goal tile
            gx, gy = -1, -1
            done = False
            for i in range(len(obs)):
                for j in range(len(obs[i])):
                    if obs[i][j][0] == OBJECT_TO_IDX["goal"]:
                        gx, gy = i, j
                        done = True
                        break
                if done: break
            self.plan = [(OraclePlanSteps.PATHFIND, (sx, sy, sr, gx, gy))]
        
        if self.env_name == "RedBlueDoors":
            # Find doors
            red_x, red_y, blue_x, blue_y = -1, -1, -1, -1
            done_red, done_blue = False, False
            for i in range(len(obs)):
                for j in range(len(obs[i])):
                    if obs[i][j][0] == OBJECT_TO_IDX["door"] and obs[i][j][1] == COLOR_TO_IDX["red"]:
                        red_x, red_y = i, j
                        done_red = True
                    elif obs[i][j][0] == OBJECT_TO_IDX["door"] and obs[i][j][1] == COLOR_TO_IDX["blue"]:
                        blue_x, blue_y = i, j
                        done_blue = True
                    if done_red and done_blue: break
                if done_red and done_blue: break
            
            # Plan
            rd_approach_x, rd_approach_y, rd_approach_r = red_x+1, red_y, 2
            bd_approach_x, bd_approach_y, bd_approach_r = blue_x-1, blue_y, 0
            self.plan = [
                            (OraclePlanSteps.PATHFIND, (sx, sy, sr, rd_approach_x, rd_approach_y, rd_approach_r)),
                            (OraclePlanSteps.OPEN),
                            (OraclePlanSteps.PATHFIND, (rd_approach_x, rd_approach_y, rd_approach_r, bd_approach_x, bd_approach_y, bd_approach_r)),
                            (OraclePlanSteps.OPEN),
            ]
        
        if self.env_name == "DoorKey":
            # Find key, door, and goal tile
            kx, ky = -1, -1
            dx, dy = -1, -1
            gx, gy = -1, -1
            done_key, done_door, done_goal = False, False, False
            for i in range(len(obs)):
                for j in range(len(obs[i])):
                    if obs[i][j][0] == OBJECT_TO_IDX["goal"]:
                        gx, gy = i, j
                        done_goal = True
                    elif obs[i][j][0] == OBJECT_TO_IDX["key"]:
                        kx, ky = i, j
                        done_key = True
                    elif obs[i][j][0] == OBJECT_TO_IDX["door"]:
                        dx, dy = i, j
                        done_door = True
                    if done_key and done_door and done_goal: break
                if done_key and done_door and done_goal: break
            
            # Evaluate 4 positions the key can be picked up from
            key_approaches = [backward_pos(kx, ky, r) for r in range(4)]
            chosen, steps = None, 1000000
            for approach in key_approaches:
                if obs[approach[0],approach[1],0] not in [OBJECT_TO_IDX["agent"], OBJECT_TO_IDX["empty"]]: continue # Check if traversable
                cost = len(self.pathfind(obs, sx, sy, sr, approach[0], approach[1], approach[2])) + \
                        len(self.pathfind(obs,approach[0], approach[1], approach[2], gx, gy, ignore_doors=True))
                if cost < steps:
                    chosen = approach
                    steps = cost
            
            door_approach_x, door_approach_y, door_approach_r = dx - 1, dy, 0
            self.plan = [
                            (OraclePlanSteps.PATHFIND, (sx, sy, sr, chosen[0], chosen[1], chosen[2])),
                            (OraclePlanSteps.PICKUP),
                            (OraclePlanSteps.PATHFIND, (chosen[0], chosen[1], chosen[2], door_approach_x, door_approach_y, door_approach_r)),
                            (OraclePlanSteps.OPEN),
                            (OraclePlanSteps.PATHFIND, (door_approach_x, door_approach_y, door_approach_r, gx, gy)),
            ]


    def forward(self, obs):
        # If no plan, then make one
        if len(self.path) == 0 and (self.plan is None or len(self.plan) == 0):
            self._plan(obs)
        
        # If no path, then we proceed with the next step of the plan
        while len(self.path) == 0:
            step = self.plan.pop(0)
            if type(step) == tuple:
                if len(step) == 1:
                    op = step[0]
                elif len(step) == 2:
                    op, args = step
            else:
                op = step
            
            if op == OraclePlanSteps.PATHFIND:
                self.path = self.pathfind(obs, *args)
            elif op == OraclePlanSteps.PICKUP:
                self.path.append(Actions.pickup)
            elif op == OraclePlanSteps.DROP:
                self.path.append(Actions.drop)
            elif op == OraclePlanSteps.OPEN:
                self.path.append(Actions.toggle)
            elif op == OraclePlanSteps.CLOSE:
                self.path.append(Actions.toggle)
            else:
                self.path.append(Actions.done)
        
        return self.path.pop(0)

        
    def reset(self):
        self.plan = None
        self.path = []
    
    def pathfind(self, obs, starting_x, starting_y, starting_r, goal_x, goal_y, goal_r = None, ignore_doors=False):
        n, m = len(obs), len(obs[0])
        starting_state = (starting_x, starting_y, starting_r)

        open_set = PriorityQueue()
        open_set.put((self.pathfind_heuristic(starting_x, starting_y, goal_x, goal_y), starting_state))

        came_from = np.zeros((n, m, 4, 3), dtype=np.int64)
        g_score = np.full((n, m, 4), n*m+n+m)
        g_score[starting_state[0], starting_state[1], starting_state[2]] = 0
        
        # f_score = np.full((n, m, 4), n*m+n+m)
        # f_score[starting_state[0], starting_state[1], starting_state[2]] = self.pathfind_heuristic(starting_state, goal_x, goal_y)

        found = False
        while open_set._qsize() > 0:
            current_pos = open_set.get()[1]
            x, y, r = current_pos[0], current_pos[1], current_pos[2]

            # Goal check
            if x == goal_x and y == goal_y and (goal_r is None or r == goal_r):
                found = True
                last_r = r
                break

            # Add neighbors, you can always rotate left or right, but we need to check if we can move forward
            neighbors = [ (x, y, (r - 1) % 4), (x, y, (r + 1) % 4) ]
            fw = forward_pos(x, y, r)
            if (-1 < fw[0] < n) and (-1 < fw[1] < m):
                world_object = obs[fw[0]][fw[1]]
                if world_object[0] in [OBJECT_TO_IDX["empty"], OBJECT_TO_IDX["goal"]] or (world_object[0] == OBJECT_TO_IDX["door"] and world_object[2] == STATE_TO_IDX["open"]) or (ignore_doors and world_object[0] in [OBJECT_TO_IDX["key"], OBJECT_TO_IDX["door"]]):
                    neighbors.append(fw)

            for (nx, ny, nr) in neighbors:
                tentative_g_score = g_score[x][y][r] + 1
                if tentative_g_score < g_score[nx][ny][nr]:
                    came_from[nx][ny][nr] = [x, y, r]
                    g_score[nx][ny][nr] = tentative_g_score
                    f_score = tentative_g_score + self.pathfind_heuristic(nx, ny, goal_x, goal_y)
                    # Check if item is in queue
                    for i, (_, pos) in enumerate(open_set.queue):
                        if pos == (nx, ny, nr):
                            open_set.queue.pop(i)
                            heapify(open_set.queue)
                    open_set.put( (f_score, (nx, ny, nr)) )
        
        path = []
        if found:
            # Reconstruct the path
            x, y, r = goal_x, goal_y, last_r

            while (x, y, r) != starting_state:
                prev_x, prev_y, prev_r = came_from[x][y][r]
                
                # Figure out the difference
                if r == prev_r: # If same rotation, it was forward
                    path.append(Actions.forward)
                elif (r - prev_r) % 4 == 1: # Right
                    path.append(Actions.right)
                elif (r - prev_r) % 4 == 3: # Left
                    path.append(Actions.left)
                else: # Shouldn't happen
                    print((x, y, r), (prev_x, prev_y, prev_r))
                    assert(False)
                
                x, y, r = prev_x, prev_y, prev_r

            path.reverse()
        
        return path
    
    # Manhattan distance between two points
    # +1 step if there is at least one direction change
    # TODO Increase estimated cost if rotation is not favorable. e.x. facing up and agent needs to go down and left
    def pathfind_heuristic(self, sx, sy, gx, gy):
        xdiff = np.abs(gx-sx)
        ydiff = np.abs(gy-sy)
        cost = xdiff + ydiff
        if xdiff > 0 and ydiff > 0: cost += 1
        return cost


