import numpy as np
import os
import pandas as pd

maps = ["Empty-16x16", "DoorKey-8x8", "RedBlueDoors-8x8", "FourRooms", "LavaCrossingS11N5", "MultiRoom-N4-S5"] 
seeds = [1,2,3,4,5,6,7,8,9,10]
dir = ""

for map in maps:
    rewards = []
    for s in seeds:
        path = os.path.join(dir, f"positions-oracle-{map}-fixed{s}.csv")
        df = pd.read_csv(path)

        for i, row in df.iterrows():
            if row["reward"] > 0:
                rewards.append(row["reward"])
    
    mean, std, med = np.mean(rewards), np.std(rewards), np.median(rewards)
    stdd = max(0.002, std)
    print(f"{map:30s}\tavg:{mean:1.4f}±{std:1.4f}\tmed:{med:1.4f}\tthres:{mean-3.5*stdd:1.4f}")
