import click
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import math

map_dims = {
    "Empty-16x16" : (16, 16, 1/16, 1/16),
    "DoorKey-8x8" : (8, 8, 1/8, 1/8),
    "RedBlueDoors-8x8" : (8, 8, 1/8, 1/8),
    "FourRooms" : (19, 19, 1/19, 1/19),
    "LavaCrossingS11N5" : (11, 11, 1/11, 1/11),
    "MultiRoom-N4-S5" : (25, 25, 1/25, 1/25),
}

snap_dict = {
    "Empty-16x16" : [250,500,1250,2500,5000],
    "DoorKey-8x8" : [500,1000,2500,5000,10000],
    "RedBlueDoors-8x8" : [500,1000,2500,5000,10000],
    "FourRooms" : [1250,2500,6250,12500,25000],
    "LavaCrossingS11N5" : [500,1000,2500,5000,10000],
    "MultiRoom-N4-S5" : [500,1000,2500,5000,10000],
}

def draw_heatmap(data, **kwargs):
    reward = data.iloc[0]["reward"]
    data = data.drop(["im", 'map'], axis=1).iloc[0]["data"]
    ax = sns.heatmap(data, square=True, cbar=False, **kwargs)
    ax.text(0.5*len(data[0]),1.1*len(data), f"{reward:3.4f}", fontsize=20, color="black", ha="center", va="center")

def draw_diff_heatmap(data, **kwargs):
    divergence = data.iloc[0]["divergence"]
    data = data.drop(["im", 'map'], axis=1).iloc[0]["data"]
    ax = sns.heatmap(data, square=True, cbar=False, cmap="vlag", **kwargs)
    ax.text(len(data[0])/2,1.1*len(data), f"{100*divergence:3.2f}%", fontsize=20, color="black", ha="center", va="center")

def zoom(data, factor):
    new_array = np.zeros((len(data)*factor, len(data[0])*factor))
    for i in range(len(new_array)):
        for j in range(len(new_array[0])):
            new_array[i][j] = data[i//factor][j//factor]
    return new_array

def sample_final_heatmaps():
    szoom = 125
    maps = ["Empty-16x16", "DoorKey-8x8", "RedBlueDoors-8x8", "FourRooms", "LavaCrossingS11N5", "MultiRoom-N4-S5"]
    seeds = [1, 8, 3, 8, 6, 9]
    im = ["nors+nomodel", "nors+statecount", "grm+statecount", "adopes+statecount",  "pies+statecount"]
    im_name = ["No IM", "State Count", "GRM", "ADOPS", "PIES"]

    big_map = []
    diff_map = []
    # Load all data
    for map, seed in zip(maps, seeds):
        max_steps = snap_dict[map][-1]
        map_width, map_height, max_v, max_diff_v = map_dims[map]
        df = pd.read_csv(f"positions-{map}-fixed{seed}.csv")
        oracle_df = pd.read_csv(f"positions-oracle-{map}-fixed{seed}.csv")
        df["im"] = df["im"].replace(im, im_name)
        df = df[df["snapshot"] == max_steps]

        base = 0
        if "RedBlueDoors" in map:
            base = 4
        
        # Oracle
        sub_df = oracle_df[["x","y"]]
        counts = sub_df.value_counts()
        oracle_sum_df = np.array([ [0.0 if (x,y) not in counts.index else counts[x,y] for x in range(base,map_width+base)] for y in range(map_height) ])
        oracle_sum_df /= np.sum(oracle_sum_df)
        avg_rew = np.mean( oracle_df.loc[ (oracle_df["done"] == True) ][["reward"]] )
        if math.isnan(avg_rew): avg_rew = 0

        sum_df = np.clip(oracle_sum_df, 0, max_v) / max_v
        factor = min(szoom//map_height, szoom//map_width)
        sum_df = zoom(sum_df, factor)
        big_map.append( (map, "A*", sum_df, avg_rew) )

        for imm in im_name:
            sub_df = df.loc[ (df["im"]==imm)][["x","y"]]
            counts = sub_df.value_counts()
            sum_df = np.array([ [0.0 if (x,y) not in counts.index else counts[x,y] for x in range(base,map_width+base)] for y in range(map_height) ])
            sum_df /= np.sum(sum_df)
            avg_rew = np.mean( df.loc[ (df["im"]==imm) & (df["done"] == True) ][["reward"]] )
            if math.isnan(avg_rew): avg_rew = 0
            diff_df = sum_df - oracle_sum_df
            diverge = np.sum(np.abs(diff_df))/2

            sum_df = np.clip(sum_df, 0, max_v) / max_v
            diff_df = np.clip(diff_df, -max_diff_v, max_diff_v) / max_diff_v
            factor = min(szoom//map_height, szoom//map_width)
            sum_df = zoom(sum_df, factor)
            diff_df = zoom(diff_df, factor)
            big_map.append( (map, imm, sum_df, avg_rew) )
            diff_map.append( (map, imm, diff_df, diverge) )

    # Heatmap
    sns.set(font_scale=1.5)
    big_map = pd.DataFrame(big_map, columns=["map", "im", "data", "reward"])
    g = sns.FacetGrid(big_map, col="im", row="map", margin_titles=True)
    superheat=g.map_dataframe(draw_heatmap, annot=False)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    for (row_val, col_val), ax in g.axes_dict.items():
        ax.set_axis_off()
    g.tight_layout()
    superheat.figure.savefig(f"final-heat.png")
    # plt.show()

    # Diff heatmap
    sns.set(font_scale=1.5)
    diff_map = pd.DataFrame(diff_map, columns=["map", "im", "data", "divergence"])
    g = sns.FacetGrid(diff_map, col="im", row="map", margin_titles=True)
    superheat=g.map_dataframe(draw_diff_heatmap, annot=False, center=0)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    for (row_val, col_val), ax in g.axes_dict.items():
        ax.set_axis_off()
    g.tight_layout()
    superheat.figure.savefig(f"final-diff.png")
    # plt.show()


@click.command()
# Testing params

def main(

):
    sample_final_heatmaps()
    

if __name__ == '__main__':
    main()
