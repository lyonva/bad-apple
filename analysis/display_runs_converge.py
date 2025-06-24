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
    im = ["nors+nomodel", "nors+statecount", "nors+maxentropy", "nors+icm", "grm+statecount", "grm+maxentropy", "grm+icm"]
    im_name = ["No IM", "State Count", "Max Entropy", "ICM", "GRM+SC", "GRM+ME", "GRM+ICM"]
    df["im"] = df["im"].replace(im, im_name)
    im = im_name
    df = df.dropna(axis=0, subset=["iterations"])

    atts = ["rollout/ep_rew_mean", "rollout/ll_unique_positions", "rollout/ll_unique_states", "rollout/ep_entropy:"]
    atts_name = ["Episode Reward", "Position Coverage", "Observation Coverage", "Entropy"]
    df = df.rename(columns=dict([(x, y) for x, y in zip(atts, atts_name)]))
    # df = df.astype({"Position Coverage" : "float32", "Observation Coverage" : "float32"})
    df = df[df["iterations"] > 850] # Only last 100 or so iters

    maps = ["DoorKey-8x8", "Empty-16x16", "FourRooms", "RedBlueDoors-8x8", "DoorKey-16x16"]
    map_threshold = [0.96, 0.96, 0.57, 0.9, 0.9]

    min_seed = 1
    max_seed = 10
    
    for map, thres in zip(maps, map_threshold):
        print(f"{map:35}" + "|".join( [ f"{s:4d}" for s in range(min_seed, max_seed+1) ] ) + f"| avg")
        for im in im_name:
            avgs = []
            count = 0
            for seed in range(min_seed, max_seed+1):
                avg = df.loc[(df["map"]==map) & (df["im"]==im) & (df["seed"]==seed), "Episode Reward"].mean()
                if avg >= thres: count += 1
                avgs.append(avg)
            print(f"{im:35}" + "|".join( [ f"{a:1.2f}" for a in avgs ]) + f"|{np.mean(avgs):1.2f}" + f"\t({count}/{max_seed-min_seed+1})" )
        print()
    

@click.command()
@click.option('--file', type=str, default="logs/alldata.csv", help='CSV file with the training statistics of all models')

def main(
    file,
):
    display_runs_converge(file)
    

if __name__ == '__main__':
    main()
