import click
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# map_dims = {
#     "Empty-16x16" : (16, 16, 0.05, 0.05),
#     "DoorKey-16x16" : (16, 16, 0.02, 0.02),
#     "RedBlueDoors-8x8" : (16, 8, 0.08, 0.08),
#     "FourRooms" : (19, 19, 0.01, 0.01),
# }

map_dims = {
    "Empty-16x16" : (16, 16, 0.04, 0.02),
    "DoorKey-8x8" : (8, 8, 0.1, 0.05),
    "DoorKey-16x16" : (16, 16, 0.025, 0.0125),
    "RedBlueDoors-8x8" : (16, 8, 0.1, 0.05),
    "FourRooms" : (19, 19, 0.0075, 0.005),
}

# def draw_log_heatmap(data, vmin=0, vmax=1, **kwargs):
#     data = data.drop(["map", "im", 'snapshot'], axis=1).iloc[0]["data"]
#     # signs = np.logical_not(np.signbit(data)).astype(np.float32)*2 - 1
#     # data = np.abs(data)
#     # data = signs*np.log10(data - np.min(data) + 0.0001)
#     data = np.log10(data - np.min(data) + 0.0001)
#     sns.heatmap(data, square=True, cbar=False, vmin=np.log10(vmin+0.0001), vmax=np.log10(vmax), **kwargs)

def draw_heatmap(data, **kwargs):
    map = data["map"].iloc[0]
    map_width, map_height, max_v, max_diff_v = map_dims[map]
    data = data.drop(["map", "im", 'snapshot'], axis=1).iloc[0]["data"]
    sns.heatmap(data, square=False, cbar=False, vmin=0, vmax=max_v, **kwargs)


def draw_diff_heatmap(data, **kwargs):
    map = data["map"].iloc[0]
    map_width, map_height, max_v, max_diff_v = map_dims[map]
    data = data.drop(["map", "im", 'snapshot'], axis=1).iloc[0]["data"]
    sum = np.sum(np.abs(data))
    ax = sns.heatmap(data, square=False, cbar=False, cmap="vlag", vmin=-max_diff_v, vmax=max_diff_v, **kwargs)
    ax.text(len(data[0])/2,len(data)/2, f"{sum:3.2f}", fontsize=16, ha="center", va="center")

def make_heatmaps_by_snap(dir, baseline):
    maps = ["DoorKey-8x8", "Empty-16x16", "FourRooms", "RedBlueDoors-8x8", "DoorKey-16x16"]
    df = pd.DataFrame()
    for map in maps:
        file = os.path.join(dir, f"positions-{map}.csv")
        sdf = pd.read_csv(file)
        sdf["map"] = map
        df = pd.concat((df, sdf))
    #print(df)

    # im = ["nomodel", "statecount", "maxentropy", "rnd", "grm"]
    # im_name = ["No IM", "State Count", "Max Entropy", "RND", "GRM"]
    # im = ["nomodel", "statecount", "maxentropy", "icm", "rnd", "grm"]
    # im_name = ["No IM", "State Count", "Max Entropy", "ICM", "RND", "GRM"]
    im = ["nors+nomodel", "nors+statecount", "nors+maxentropy", "nors+icm", "grm+statecount", "grm+maxentropy", "grm+icm"]
    im_name = ["No IM", "State Count", "Max Entropy", "ICM", "GRM+SC", "GRM+ME", "GRM+ICM"]
    # im = ["nors+nomodel", "nors+statecount", "nors+maxentropy", "nors+icm", "grm+statecount", "grm+maxentropy", "grm+icm", "adopes+statecount", "adopes+maxentropy", "adopes+icm"]
    # im_name = ["No IM", "State Count", "Max Entropy", "ICM", "GRM+SC", "GRM+ME", "GRM+ICM", "ADOPES+SC", "ADOPES+ME", "ADOPES+ICM"]
    df["im"] = df["im"].replace(im, im_name)
    im = im_name

    # sn1 = [3, 15, 305, 610, 915, 1221]
    # sn_name = ["0.25%", "1.25%", "25%", "50%", "75%", "100%"]
    # sn1 = [25, 49, 123, 245, 489]
    sn1 = [50,100,250,500,1000]
    sn_name = ["5%", "10%", "25%", "50%", "100%"]
    df["snapshot"] = df["snapshot"].replace(sn1, sn_name)

    ims = im_name
    snapshots = sn_name

    big_map = []
    for snapshot in snapshots:
        for map in maps:
            map_width, map_height, max_v, max_diff_v = map_dims[map]

            # Filter by seed
            # df = df[df["seed"]==1]

            for im in ims:
                # counts = np.zeros((mapsize, mapsize))
                sub_df = df.loc[ (df["snapshot"] == snapshot) & (df["map"] == map) & (df["im"]==im)  ][["x","y"]]
                counts = sub_df.value_counts() / sub_df.shape[0]
                sum_df = np.array([ [0 if (x,y) not in counts.index else counts[x,y] for x in range(map_width)] for y in range(map_height) ])
                big_map.append( (map, snapshot, im, sum_df) )
    big_map = pd.DataFrame(big_map, columns=["map", "snapshot", "im", "data"])
    
    # Diff heatmap
    diff_map = []
    for snapshot in snapshots:
        for map in maps:
            base = big_map[(big_map["map"] == map) & (big_map["im"] == baseline) & (big_map["snapshot"] == snapshot)].drop(["map", "im", 'snapshot'], axis=1).iloc[0]["data"]
            for im in ims:
                if im == baseline: continue
                tech = big_map[(big_map["map"] == map) & (big_map["im"] == im) & (big_map["snapshot"] == snapshot)].drop(["map", "im", 'snapshot'], axis=1).iloc[0]["data"]
                tech -= base
                diff_map.append( (map, snapshot, im, tech) )
    diff_map = pd.DataFrame(diff_map, columns=["map", "snapshot", "im", "data"])


    # Plot
    for snapshot in snapshots:
        # Heatmap
        sub_big = big_map[ big_map["snapshot"] == snapshot ]
        g = sns.FacetGrid(sub_big, row="im", col="map", margin_titles=True)
        superheat=g.map_dataframe(draw_heatmap, annot=False)
        g.set_titles(col_template="Training: {col_name}", row_template="{row_name}")
        superheat.figure.savefig(f"heat-{snapshot}.png")
        # plt.show()

        # Diff Heatmap
        sub_diff = diff_map[ diff_map["snapshot"] == snapshot ]
        g = sns.FacetGrid(sub_diff, row="im", col="map", margin_titles=True)
        superheat=g.map_dataframe(draw_diff_heatmap, annot=False)
        g.set_titles(col_template="Training: {col_name}", row_template="{row_name}")
        superheat.figure.savefig(f"diff-{snapshot}.png")
        # plt.show()


@click.command()
# Testing params
@click.option('--dir', type=str, default=".", help='Dir with the CSV files with the positions of agents')
@click.option('--baseline', default="No IM", type=str, help='Name of the baseline method')

def main(
    dir, baseline,
):
    make_heatmaps_by_snap(dir, baseline)
    

if __name__ == '__main__':
    main()
