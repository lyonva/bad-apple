import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import import_config_module, get_map_snaps
from scipy.stats import ttest_1samp, sem

def display_runs_converge(file, config_file, confidence, hold, window):
    df = pd.read_csv(file)
    config = import_config_module(config_file)

    im = config.im
    im_name = config.im_name

    df["im"] = df["im"].replace(im, im_name)
    im = im_name
    df = df.dropna(axis=0, subset=["iterations"])

    # atts = ["rollout/ep_rew_mean", "rollout/ll_unique_positions", "rollout/ll_unique_states", "rollout/ep_entropy:"]
    # atts_name = ["Episode Reward", "Position Coverage", "Observation Coverage", "Entropy"]
    # df = df.rename(columns=dict([(x, y) for x, y in zip(atts, atts_name)]))
    # df = df.astype({"Position Coverage" : "float32", "Observation Coverage" : "float32"})

    # maps = ["Empty-16x16", "DoorKey-8x8", "RedBlueDoors-8x8", "FourRooms", "LavaCrossingS11N5", "MultiRoom-N4-S5"]
    # maps_length = [5000, 10000, 10000, 25000, 10000, 10000]
    # map_threshold = [0.96927, 0.96078, 0.98043, 0.66324, 0.95155, 0.69076]

    maps = config.maps
    default_snaps = config.default_snaps
    maps_snapshot = config.maps_snapshot

    maps_length = [get_map_snaps(map, maps_snapshot, default_snaps)[-1] for map in maps]
    map_threshold = config.map_threshold

    min_seed = config.seeds[0]
    max_seed = config.seeds[-1]

    # Atari: Remove iterations where no episodes end
    df = df[df["rollout/ep_len_mean"] > 0]


    # First show iteration where it converges
    print("-"*10 + "Rollout for convergence" + "-"*10)
    for map, thres, max_length in zip(maps, map_threshold, maps_length):
        print(f"{map:35}" + "|".join( [ f"{s:5d}" for s in range(min_seed, max_seed+1) ] ) + f"|                avg|                rel|count")


        for im in im_name:
            avgs = []
            steps = []
            count = 0
            for seed in range(min_seed, max_seed+1):
                sub_df = df.loc[(df["map"]==map) & (df["im"]==im) & (df["seed"]==seed), ["iterations", "rollout/ep_rew_mean"]]
                current_combo = 0
                current_idx = -1
                for i in range(window, max_length):
                    sequence = sub_df[(sub_df["iterations"]<=i) & (sub_df["iterations"]>i-window)]["rollout/ep_rew_mean"].to_numpy()
                    _, pvalue = ttest_1samp(sequence, thres, alternative='greater')

                    if pvalue/2 < (1 - confidence):
                        if current_combo == 0:
                            current_idx = i
                        current_combo += 1
                        if current_combo >= hold: break
                    else:
                        current_combo = 0
                        current_idx = -1
            
                if current_combo < hold: current_idx = -1
                steps.append(int(current_idx))

                if current_idx > 0:
                    avgs.append(current_idx)
                    count += 1
            avg = 0 if len(avgs) == 0 else np.mean(avgs)
            std = 0 if len(avgs) < 2 else sem(avgs)
            print(f"{im:35}" + "|".join( [ f"{a:5d}" for a in steps ]) + f"|{avg:>8.2f} ({std:>8.2f})" + f"|{(avg/max_length)*100:>7.2f}% ({(std/max_length)*100:>7.2f}%)" + f"|{count:3d}/{max_seed-min_seed+1:3d}" )
        print()

    print()
    print()
    print("-"*10 + "Average Final Reward" + "-"*10)
    # Iteration to percentage
    df["iterations"] = df["iterations"].astype(np.float64)
    for map, length in zip(maps, maps_length):
        df.loc[df["map"] == map, "iterations"] /= length


    df = df[df["iterations"] > 0.95] # Only last 5% training

    
    for map, thres in zip(maps, map_threshold):
        print(f"{map:35}" + "|".join( [ f"{s:6d}" for s in range(min_seed, max_seed+1) ] ) + f"|" + (" "*12) +"avg±std" + "\t(conv)")
        for im in im_name:
            avgs = []
            count = 0
            for seed in range(min_seed, max_seed+1):
                avg = df.loc[(df["map"]==map) & (df["im"]==im) & (df["seed"]==seed), "rollout/ep_rew_mean"].mean()
                if avg >= (thres): count += 1
                avgs.append(avg)
            print(f"{im:35}" + "|".join( [ f"{a:6.2f}" for a in avgs ]) + f"|{np.mean(avgs):9.4f}±{np.std(avgs):>9.4f}" + f"\t({count}/{max_seed-min_seed+1})" )
        print()
    

@click.command()
@click.option('--file', type=str, default="logs/alldata.csv", help='CSV file with the training statistics of all models')
@click.option('--config', default='config', type=str, help='Config file')
@click.option('--confidence', default=0.95, type=float, help='Test confidence level')
@click.option('--hold', default=50, help="Number of training episodes for which the agent must hold the statistical significance test")
@click.option('--window', default=100, help='Number of most recent data points to use in test')

def main(
    file, config, confidence, hold, window
):
    display_runs_converge(file, config, confidence, hold, window)
    

if __name__ == '__main__':
    main()
