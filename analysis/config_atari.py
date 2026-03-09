default_snaps = [1000, 2000, 3000, 4000]

parameters = []

maps = ["Pong", "Asteroids", "Freeway", "Boxing", "Bowling", "FishingDerby", "Seaquest"]
maps_name = ["Pong", "Asteroids", "Freeway", "Boxing", "Bowling", "FishingDerby", "Seaquest"]
maps_snapshot = {}
map_threshold = [0, 18100, 32, 15, 149, 20, 101120]
# map_threshold = [15, 47389, 30, 12, 161, -39]

im = ["NoRS+NoModel", "NoRS+RND", "GRM+RND", "ADOPS+RND"]
im_name = ["PPO", "RND", "GRM", "ADOPS"]

ticks_dict = {
    "Asteroids" : [1000,2000,3000,4000],
    "Freeway" : [1000,2000,3000,4000],
    "Pong" : [1000,2000,3000,4000],
    "Boxing" : [1000,2000,3000,4000],
    "Bowling" : [1000,2000,3000,4000],
    "FishingDerby" : [1000,2000,3000,4000],
}


metrics = ["rollout/ep_rew_mean", "rollout/ep_len_mean"]
metrics_name = ["Average Episode Reward", "Average Episode Length"]

seeds = [1,2,3,4,5]
