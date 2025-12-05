import click
import glob
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

map_dims = {
    "Empty-16x16" : (16, 16, 0.04, 0.02),
    "DoorKey-8x8" : (8, 8, 0.075, 0.0375),
    "DoorKey-16x16" : (16, 16, 0.025, 0.0125),
    "RedBlueDoors-8x8" : (16, 8, 0.1, 0.05),
    "FourRooms" : (19, 19, 0.01, 0.02),
    "LavaCrossingS11N5" : (11, 11, 0.05, 0.025),
    "MultiRoom-N4-S5" : (25, 25, 0.025, 0.0125),
}

snap_dict = {
    "Empty-16x16" : [250,500,1250,2500,5000],
    "DoorKey-8x8" : [500,1000,2500,5000,10000],
    "RedBlueDoors-8x8" : [500,1000,2500,5000,10000],
    "FourRooms" : [1250,2500,6250,12500,25000],
    "LavaCrossingS11N5" : [500,1000,2500,5000,10000],
    "MultiRoom-N4-S5" : [500,1000,2500,5000,10000],
}

def calculate_divergence(dir):
    maps = ["Empty-16x16", "DoorKey-8x8", "RedBlueDoors-8x8", "FourRooms", "LavaCrossingS11N5", "MultiRoom-N4-S5"]
    # maps = ["Empty-16x16", "DoorKey-8x8", "FourRooms"]
    seeds = [1,2,3,4,5,6,7,8,9,10]
    imm = ["nors+nomodel", "nors+statecount", "grm+statecount", "adopes+statecount",  "pies+statecount"]
    im_name = ["No IM", "State Count", "GRM", "ADOPS", "PIES"]

    big_map = []

    for map in maps:
        max_steps = snap_dict[map][-1]
        map_width, map_height, max_v, max_diff_v = map_dims[map]

        for seed in seeds:
            if seed > 1 and "Empty" in map: break
            # print(os.path.join(dir, f"positions-{map}-fixed{seed}.csv"))
            df = pd.read_csv(os.path.join(dir, f"positions-{map}-fixed{seed}.csv"))
            df["im"] = df["im"].replace(imm, im_name)
            df = df[df["snapshot"] == max_steps]
            oracle_df = pd.read_csv(os.path.join(dir, f"positions-oracle-{map}-fixed{seed}.csv"))
            
            sub_df = oracle_df[["x","y"]]
            counts = sub_df.value_counts()
            oracle_policy = np.array([ [0.0 if (x,y) not in counts.index else counts[x,y] for x in range(map_width)] for y in range(map_height) ])
            oracle_policy /= np.sum(oracle_policy)

            for im in im_name:
                sub_df = df.loc[ (df["im"]==im)][["x","y"]]
                counts = sub_df.value_counts()
                policy = np.array([ [0.0 if (x,y) not in counts.index else counts[x,y] for x in range(map_width)] for y in range(map_height) ])
                policy /= np.sum(policy)
                divergence = np.sum(np.abs(policy - oracle_policy)) / 2

                big_map.append([map, seed, im, divergence])
    
    big_map = pd.DataFrame(big_map, columns=["map", "seed", "im", "divergence"])
    
    for map in maps:
        print(f"{map}")
        for im in im_name:
            data = big_map.loc[ (big_map["map"] == map) & (big_map["im"] == im) ]["divergence"]
            print(f"{im:15s}: {data.mean()*100:6.2f}% ± {data.std()*100:6.2f}%")
        print()

@click.command()
# Testing params
@click.option('--dir', type=str, default=".", help='Directory with the csv files with the positions of the agents')

def main(
    dir
):
    calculate_divergence(dir)
    

if __name__ == '__main__':
    main()
