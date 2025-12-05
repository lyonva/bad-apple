import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def display_runs_converge(file):
    # sns.set_theme(style="darkgrid")
    sns.set_theme(style="ticks", rc={'font.family':'serif', 'font.serif':'Times New Roman'})

    df = pd.read_csv(file)
    # im = ["nomodel", "statecount", "grm"]
    # im_name = ["No IM", "State Count", "GRM"]
    # im = ["nomodel", "statecount", "maxentropy", "icm", "rnd", "grm"]
    # im_name = ["No IM", "State Count", "Max Entropy", "ICM", "RND", "GRM"]
    # im = ["nors+nomodel", "nors+statecount", "nors+maxentropy", "nors+icm", "grm+statecount", "grm+maxentropy", "grm+icm", "adopes+statecount", "adopes+maxentropy", "adopes+icm"]
    # im_name = ["No IM", "State Count", "Max Entropy", "ICM", "GRM+SC", "GRM+ME", "GRM+ICM", "ADOPES+SC", "ADOPES+ME", "ADOPES+ICM"]
    # im = ["nors+nomodel", "nors+statecount", "nors+maxentropy", "nors+icm", "grm+statecount", "grm+maxentropy", "grm+icm"]
    # im_name = ["No IM", "State Count", "Max Entropy", "ICM", "GRM+SC", "GRM+ME", "GRM+ICM"]
    im = ["nors+nomodel", "nors+statecount", "grm+statecount", "adopes+statecount",  "pies+statecount"]
    im_name = ["No IM", "State Count", "GRM+SC", "ADOPES+SC", "PIES+SC"]
    df["im"] = df["im"].replace(im, im_name)
    im = im_name
    df = df.dropna(axis=0, subset=["iterations"])

    atts = ["rollout/ep_rew_mean", "rollout/ll_unique_positions", "rollout/ll_unique_states", "rollout/ep_entropy:"]
    atts_name = ["Episode Reward", "Position Coverage", "Observation Coverage", "Entropy"]
    df = df.rename(columns=dict([(x, y) for x, y in zip(atts, atts_name)]))
    # df = df.astype({"Position Coverage" : "float32", "Observation Coverage" : "float32"})

    maps = ["Empty-16x16", "DoorKey-8x8", "RedBlueDoors-8x8", "FourRooms", "LavaCrossingS11N5", "MultiRoom-N4-S5"]
    maps_length = [5000, 10000, 10000, 25000, 10000, 10000]
    map_threshold = [0.96927, 0.96078, 0.98043, 0.66324, 0.95155, 0.69076]

    min_seed = 1
    max_seed = 10


    # First show iteration where it converges
    max_combo = 25

    print("-"*10 + "Rollout for convergence" + "-"*10)
    for map, thres in zip(maps, map_threshold):
        print(f"{map:35}" + "|".join( [ f"{s:5d}" for s in range(min_seed, max_seed+1) ] ) + f"|               avg|count")

        for im in im_name:
            avgs = []
            steps = []
            count = 0
            for seed in range(min_seed, max_seed+1):
                sub_df = df.loc[(df["map"]==map) & (df["im"]==im) & (df["seed"]==seed), ["iterations", "Episode Reward"]]
                current_combo = 0
                current_idx = 0
                for i, row in sub_df.iterrows():
                    if row["Episode Reward"] > thres:
                        if current_idx == 0: current_idx = row["iterations"]
                        current_combo += 1
                        if current_combo >= max_combo: break
                    else:
                        current_combo = current_idx = 0
                if current_combo < max_combo: current_idx = 0
                steps.append(int(current_idx))
                if current_idx > 0:
                    avgs.append(current_idx)
                    count += 1
            avg = 0 if len(avgs) == 0 else np.mean(avgs)
            std = 0 if len(avgs) < 2 else np.std(avgs)
            print(f"{im:35}" + "|".join( [ f"{a:5d}" for a in steps ]) + f"|{avg:>8.2f} ±{std:>8.2f}" + f"|{count:3d}/{max_seed-min_seed+1:3d}" )
        print()

    print()
    print()
    print("-"*10 + "Average Final Reward" + "-"*10)
    # Iteration to percentage
    df["iterations"] = df["iterations"].astype(np.float64)
    for map, length in zip(maps, maps_length):
        df.loc[df["map"] == map, "iterations"] /= length


    df = df[df["iterations"] > 0.90] # Only last 5% training

    
    for map, thres in zip(maps, map_threshold):
        print(f"{map:35}" + "|".join( [ f"{s:4d}" for s in range(min_seed, max_seed+1) ] ) + f"|            avg")
        for im in im_name:
            avgs = []
            count = 0
            for seed in range(min_seed, max_seed+1):
                avg = df.loc[(df["map"]==map) & (df["im"]==im) & (df["seed"]==seed), "Episode Reward"].mean()
                if avg >= thres: count += 1
                avgs.append(avg)
            print(f"{im:35}" + "|".join( [ f"{a:1.2f}" for a in avgs ]) + f"|{np.mean(avgs):7.4f}±{np.std(avgs):>7.4f}" + f"\t({count}/{max_seed-min_seed+1})" )
        print()
    

@click.command()
@click.option('--file', type=str, default="logs/alldata.csv", help='CSV file with the training statistics of all models')

def main(
    file,
):
    display_runs_converge(file)
    

if __name__ == '__main__':
    main()
