import click
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import math
from os.path import join

map_dims = {
    "Empty-16x16" : (16, 16, 1/14/14, 0.02),
    "DoorKey-8x8" : (8, 8, 1/6/6, 0.0375),
    "RedBlueDoors-8x8" : (8, 8, 1/6/6, 0.05),
    "FourRooms" : (19, 19, 1/17/17, 0.02),
    "LavaCrossingS11N5" : (11, 11, 1/9/9, 0.025),
    "MultiRoom-N4-S5" : (25, 25, 1/23/23, 0.0125),
}

snap_dict = {
    "Empty-16x16" : [3,7,13,25,50,100,150,200,250],
    "DoorKey-8x8" : [5,13,25,50,100,200,300,400,500],
    "RedBlueDoors-8x8" : [5,13,25,50,100,200,300,400,500],
    "FourRooms" : [13,32,63,125,250,500,750,1000,1250],
    "LavaCrossingS11N5" : [5,13,25,50,100,200,300,400,500],
    "MultiRoom-N4-S5" : [5,13,25,50,100,200,300,400,500],
}

# snap_dict = {
#     "Empty-16x16" : [3,7,13,25],
#     "DoorKey-8x8" : [5,13,25,50],
#     "RedBlueDoors-8x8" : [5,13,25,50],
#     "FourRooms" : [13,32,63,125],
#     "LavaCrossingS11N5" : [5,13,25,50],
#     "MultiRoom-N4-S5" : [5,13,25,50],
# }

def zoom(data, factor):
    new_array = np.zeros((len(data)*factor, len(data[0])*factor))
    for i in range(len(new_array)):
        for j in range(len(new_array[0])):
            new_array[i][j] = data[i//factor][j//factor]
    return new_array

def draw_heatmap(data, **kwargs):
    ex_rate = data.iloc[0]["exp rate"]
    data = data.drop(["im", 'map'], axis=1).iloc[0]["data"]
    ax = sns.heatmap(data, square=True, cbar=False, annot=False, **kwargs)
    ax.text(len(data[0])//2,1.1*len(data), f"{ex_rate*100:3.2f}%", fontsize=22, color="black", ha="center", va="center")

def make_heatmaps_exploration(dir):
    maps = ["Empty-16x16", "DoorKey-8x8", "RedBlueDoors-8x8", "FourRooms", "LavaCrossingS11N5", "MultiRoom-N4-S5"]
    im_base = ["nors+nomodel", "nors+statecount", "grm+statecount", "adopes+statecount",  "pies+statecount"]
    im_name = ["No IM", "State Count", "GRM", "ADOPES", "PIES"]

    big_map = []

    for map in maps:
        snaps = snap_dict[map]
        map_width, map_height, max_v, max_diff_v = map_dims[map]
        seeds = [1] if "Empty" in map else [1,2,3,4,5,6,7,8,9,10]
        # seeds = [1] if "Empty" in map else [1]
        tech_map = {}

        for im in im_name:
            tech_map[im] = np.zeros((map_height, map_width))
 
        for seed in seeds:
            file = join(dir, f"positions-{map}-fixed{seed}.csv")
            df = pd.read_csv(file)
            df["im"] = df["im"].replace(im_base, im_name)

            for im in im_name:
                for snap in snaps:
                    sub_df = df.loc[ (df["im"]==im) & (df["snapshot"] == snap) ][["x","y"]]
                    counts = sub_df.value_counts()
                    base = 0
                    if "RedBlueDoors" in map:
                        base = 4
                    sum_df = np.array([ [0.0 if (x,y) not in counts.index else counts[x,y] for x in range(base,base+map_width)] for y in range(map_height) ])
                    sum_df /= np.sum(sum_df)
                    tech_map[im] += sum_df
            
        for im in im_name:
            tech = tech_map[im]
            tech /= len(seeds) * len(snaps)
            tech = np.clip(tech, 0, max_v) / max_v
            ex_rate = np.sum(tech)/((map_height-2)*(map_width-2))
            szoom = 125
            factor = min(szoom//map_height, szoom//map_width)
            tech = zoom(tech, factor)
            big_map.append([map, im, tech, ex_rate])
            
    big_map = pd.DataFrame(big_map, columns=["map", "im", "data", "exp rate"])

    # Heatmap
    sns.set(font_scale=1.5)
    g = sns.FacetGrid(big_map, col="im", row="map", margin_titles=True)
    superheat=g.map_dataframe(draw_heatmap)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    for (row_val, col_val), ax in g.axes_dict.items():
        ax.set_axis_off()
        ax.set_aspect('equal','box')
    g.tight_layout()
    superheat.figure.savefig(f"heat-exploration.png")
    # plt.show()


@click.command()
# Testing params
@click.option('--dir', type=str, default=".", help='Directory with the csv files with the positions of the agents')

def main(
    dir, 
):
    make_heatmaps_exploration(dir)
    

if __name__ == '__main__':
    main()
