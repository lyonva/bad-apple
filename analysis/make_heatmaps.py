import click
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import math
from utils import import_config_module, get_map_snaps

def draw_log_heatmap(data, vmin=0, vmax=1, **kwargs):
    data = data.drop(["im", 'snapshot'], axis=1).iloc[0]["data"]
    # signs = np.logical_not(np.signbit(data)).astype(np.float32)*2 - 1
    # data = np.abs(data)
    # data = signs*np.log10(data - np.min(data) + 0.0001)
    data = np.log10(data - np.min(data) + 0.0001)
    sns.heatmap(data, square=True, cbar=False, vmin=np.log10(vmin+0.0001), vmax=np.log10(vmax), **kwargs)

def draw_heatmap(data, **kwargs):
    reward = data.iloc[0]["reward"]
    # cost = data.iloc[0]["cost"]
    data = data.drop(["im", 'snapshot'], axis=1).iloc[0]["data"]
    ax = sns.heatmap(data, square=True, cbar=False, **kwargs)
    # ax.text(3,len(data)-1, f"{reward:3.5f}", fontsize=16, color="white", ha="center", va="center")
    # ax.text(len(data),len(data)-1, f"{cost}", fontsize=16, color="red", ha="right", va="center")
    ax.text(len(data[0])//2,1.1*len(data), f"{reward:5.4f}", fontsize=22, color="black", ha="center", va="center")


def draw_diff_heatmap(data, **kwargs):
    data = data.drop(["im", 'snapshot'], axis=1).iloc[0]["data"]
    sum = np.sum(np.abs(data))
    ax = sns.heatmap(data, square=True, cbar=False, cmap="vlag", **kwargs)
    ax.text(len(data[0])/2,len(data)/2, f"{sum:3.2f}", fontsize=16, ha="center", va="center")

def make_heatmaps(file, config_file, aggregate, exploration):
    df = pd.read_csv(file)
    config = import_config_module(config_file)

    im = config.im
    im_name = config.im_name
    maps = config.maps
    maps_name = config.maps_name

    df = df.replace({"map" : dict([(x, y) for x, y in zip(maps, maps_name)]), "im" : dict([(x, y) for x, y in zip(im, im_name)])})
    
    ims = im_name
    maps = maps_name

    seeds = config.seeds
    
    for map in maps:
        map_df = df[df["map"]==map]

        map_width, map_height, max_v, max_diff_v = config.map_dims[map]
        snap = get_map_snaps(map, config.maps_snapshot, config.default_snaps)
        snapshots = [f"{100*(s/snap[-1])}%" for s in snap]
        map_df.loc[:,"snapshot"] = map_df.loc[:,"snapshot"].replace(snap, snapshots)
        
        for seed in seeds:
            seed_df = map_df[(map_df["map"]==map) & (map_df["seed"]==seed)]
            big_map = []

            for im in ims:
                if aggregate > 0:
                    count = 0
                    all_sum_df = None
                    all_avg_rew = []
                    # all_costs = []
                for snapshot in snapshots:
                    # counts = np.zeros((mapsize, mapsize))
                    sub_df = seed_df.loc[ (seed_df["im"]==im) & (seed_df["snapshot"] == snapshot) ][["x","y"]]
                    counts = sub_df.value_counts() / sub_df.shape[0]
                    sum_df = np.array([ [0.0 if (x,y) not in counts.index else counts[x,y] for x in range(map_width)] for y in range(map_height) ])
                    avg_rew = np.mean( seed_df.loc[ (seed_df["im"]==im) & (seed_df["snapshot"] == snapshot) & (seed_df["done"] == True) ][["reward"]] )
                    if math.isnan(avg_rew): avg_rew = 0
                    total_cost = np.sum( df.loc[ (df["im"]==im) & (df["snapshot"] == snapshot) ][["cost"]], axis=0 ).item()
                    # big_map.append( (snapshot, im, sum_df, avg_rew) )
                    big_map.append( (snapshot, im, sum_df, avg_rew, total_cost) )
                    if aggregate > 0:
                        count += 1
                        if all_sum_df is None:
                            all_sum_df = sum_df.copy()
                        else:
                            all_sum_df += sum_df
                        all_avg_rew.append(avg_rew)
                if aggregate:
                    big_map.append( ("Cummulative", im, all_sum_df, np.mean(all_avg_rew)) )
    

            # Heatmap
            sns.set(font_scale=1.5)
            big_map = pd.DataFrame(big_map, columns=["snapshot", "im", "data", "reward", "cost"])
            g = sns.FacetGrid(big_map, col="im", row="snapshot", margin_titles=True)
            superheat=g.map_dataframe(draw_heatmap, annot=False, vmin=0, vmax=max_v)
            g.set_titles(col_template="{col_name}", row_template="Training: {row_name}")
            for (row_val, col_val), ax in g.axes_dict.items():
                ax.set_axis_off()
            seed = "" if seed is None else f"-fixed{seed}"
            superheat.figure.savefig(f"heat-{map}{seed}.png")
    # plt.show()

    # Diff heatmap
    # diff_map = []
    # for snapshot in snapshots:
    #     base = big_map[(big_map["im"] == baseline) & (big_map["snapshot"] == snapshot)].drop(["im", 'snapshot'], axis=1).iloc[0]["data"]
    #     for im in ims:
    #         if im == baseline: continue
    #         tech = big_map[(big_map["im"] == im) & (big_map["snapshot"] == snapshot)].drop(["im", 'snapshot'], axis=1).iloc[0]["data"]
    #         tech -= base
    #         diff_map.append( (snapshot, im, tech) )
    # diff_map = []
    # for snapshot in snapshots:
    #     base = big_map[(big_map["im"] == baseline) & (big_map["snapshot"] == snapshot)].drop(["im", 'snapshot'], axis=1).iloc[0]["data"]
    #     for im in ims:
    #         if im == baseline: continue
    #         tech = big_map[(big_map["im"] == im) & (big_map["snapshot"] == snapshot)].drop(["im", 'snapshot'], axis=1).iloc[0]["data"]
    #         tech -= base
    #         diff_map.append( (snapshot, im, tech) )
    
    # sns.set(font_scale=1.5)
    # diff_map = pd.DataFrame(diff_map, columns=["snapshot", "im", "data"])
    # g = sns.FacetGrid(diff_map, col="im", row="snapshot", margin_titles=True)
    # superheat=g.map_dataframe(draw_diff_heatmap, annot=False, vmin=-max_diff_v, vmax=max_diff_v)
    # g.set_titles(col_template="Training: {col_name}", row_template="{row_name}")
    # for (row_val, col_val), ax in g.axes_dict.items():
    #     ax.set_axis_off()
    # superheat.figure.savefig(f"diff-{map}{seed}.png")
    # sns.set(font_scale=1.5)
    # diff_map = pd.DataFrame(diff_map, columns=["snapshot", "im", "data"])
    # g = sns.FacetGrid(diff_map, col="im", row="snapshot", margin_titles=True)
    # superheat=g.map_dataframe(draw_diff_heatmap, annot=False, vmin=-max_diff_v, vmax=max_diff_v)
    # g.set_titles(col_template="Training: {col_name}", row_template="{row_name}")
    # for (row_val, col_val), ax in g.axes_dict.items():
    #     ax.set_axis_off()
    # superheat.figure.savefig(f"diff-{map}{seed}.png")
    # plt.show()


@click.command()
# Testing params
@click.option('--file', default="positions.csv", type=str, help='CSV file with the positions of agents')
@click.option('--config', default='config', type=str, help='Config file')
@click.option('--aggregate', default=0, type=int, help='Whether to do a final heatmap that combines all snapshots')
@click.option('--exploration', default=0, type=int, help='Whether to display the exploration rate')

def main(
    file, config, aggregate, exploration
):
    make_heatmaps(file, config, aggregate, exploration)
    

if __name__ == '__main__':
    main()
