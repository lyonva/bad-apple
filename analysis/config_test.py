env_source = "MiniGrid"

default_snaps = [250,500,1250,2500,5000]

parameters = []

maps = ["EmptyCenter-35x35", "CliffWalk-12x9-S25"]
maps_name = ["EmptyCenter", "CliffWalk"]
map_threshold = [0.97, 0.9]

im = ["NoRS+NoModel", "NoRS+StateCount", "GRM+StateCount", "ADOPS+StateCount"]
im_name = ["PPO", "IM", "GRM", "ADOPS"]

maps_snapshot = {
    "Empty" : [250,500,1250,2500,5000],
    "DoorKey" : [500,1000,2500,5000,10000],
    "RedBlueDoors" : [500,1000,2500,5000,10000],
    "FourRooms" : [1250,2500,6250,12500,25000],
    "LavaCrossing" : [500,1000,2500,5000,10000],
    "MultiRoom" : [500,1000,2500,5000,10000],
    "EmptyCenter" : [25,100,250,500,1000,10000],
    "CliffWalk" : [25,100,250,500,1000,10000],
}

ticks_dict = {
    "Empty" : [0,1000,2000,3000,4000,5000],
    "DoorKey" : [0,2000,4000,6000,8000,10000],
    "RedBlueDoors" : [0,2000,4000,6000,8000,10000],
    "FourRooms" : [0,5000,10000,15000,20000,25000],
    "LavaCrossing" : [0,2000,4000,6000,8000,10000],
    "MultiRoom" : [0,2000,4000,6000,8000,10000],
    "EmptyCenter" : [0,2000,4000,6000,8000,10000],
    "CliffWalk" : [0,2000,4000,6000,8000,10000],
}

map_dims = {
    "Empty" : (16, 16, 0.04, 0.02),
    "DoorKey" : (8, 8, 0.075, 0.0375),
    "DoorKey" : (16, 16, 0.025, 0.0125),
    "RedBlueDoors" : (16, 8, 0.1, 0.05),
    "FourRooms" : (19, 19, 0.01, 0.02),
    "LavaCrossing" : (11, 11, 0.05, 0.025),
    "MultiRoom" : (25, 25, 0.025, 0.0125),
    "EmptyCenter" : [35, 35, 1e-3, 1e-4],
    "CliffWalk" : [12, 9, 1e-3, 5e-4],
}

metrics = ["rollout/ep_rew_mean"]
metrics_name = ["Average Episode Reward"]

seeds = [1,2,3]
