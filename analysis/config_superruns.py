# 

default_snaps = [250,500,1250,2500,5000]

parameters = []

maps = ["Empty-16x16", "DoorKey-8x8", "RedBlueDoors-8x8", "FourRooms", "LavaCrossingS11N5", "MultiRoom-N4-S5"]
maps_name = ["Empty", "DoorKey", "RedBlueDoors", "FourRooms", "LavaCrossing", "MultiRoom"]
map_threshold = [0.9693, 0.9608, 0.9804, 0.6632, 0.9515, 0.6908]

im = ["nors+nomodel", "nors+statecount", "grm+statecount", "adopes+statecount"]
im_name = ["PPO", "IM", "GRM", "ADOPS"]

maps_snapshot = {
    "Empty" : [250,500,1250,2500,5000],
    "DoorKey" : [500,1000,2500,5000,10000],
    "RedBlueDoors" : [500,1000,2500,5000,10000],
    "FourRooms" : [1250,2500,6250,12500,25000],
    "LavaCrossing" : [500,1000,2500,5000,10000],
    "MultiRoom" : [500,1000,2500,5000,10000],
}

ticks_dict = {
    "Empty" : [0,1000,2000,3000,4000,5000],
    "DoorKey" : [0,2000,4000,6000,8000,10000],
    "RedBlueDoors" : [0,2000,4000,6000,8000,10000],
    "FourRooms" : [0,5000,10000,15000,20000,25000],
    "LavaCrossing" : [0,2000,4000,6000,8000,10000],
    "MultiRoom" : [0,2000,4000,6000,8000,10000],
}

metrics = ["rollout/ep_rew_mean"]
metrics_name = ["Average Episode Reward"]

seeds = [1,2,3,4,5,6,7,8,9,10]
