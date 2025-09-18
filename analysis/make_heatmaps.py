import click
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

# map_dims = {
#     "Empty-16x16" : (16, 16, 0.05, 0.05),
#     "DoorKey-16x16" : (16, 16, 0.02, 0.02),
#     "RedBlueDoors-8x8" : (16, 8, 0.08, 0.08),
#     "FourRooms" : (19, 19, 0.01, 0.01),
# }

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

def draw_log_heatmap(data, vmin=0, vmax=1, **kwargs):
    data = data.drop(["im", 'snapshot'], axis=1).iloc[0]["data"]
    # signs = np.logical_not(np.signbit(data)).astype(np.float32)*2 - 1
    # data = np.abs(data)
    # data = signs*np.log10(data - np.min(data) + 0.0001)
    data = np.log10(data - np.min(data) + 0.0001)
    sns.heatmap(data, square=True, cbar=False, vmin=np.log10(vmin+0.0001), vmax=np.log10(vmax), **kwargs)

def draw_heatmap(data, **kwargs):
    reward = data.iloc[0]["reward"]
    cost = data.iloc[0]["cost"]
    data = data.drop(["im", 'snapshot'], axis=1).iloc[0]["data"]
    ax = sns.heatmap(data, square=True, cbar=False, **kwargs)
    ax.text(3,len(data)-1, f"{reward:3.5f}", fontsize=16, color="white", ha="center", va="center")
    ax.text(len(data),len(data)-1, f"{cost}", fontsize=16, color="red", ha="right", va="center")


def draw_diff_heatmap(data, **kwargs):
    data = data.drop(["im", 'snapshot'], axis=1).iloc[0]["data"]
    sum = np.sum(np.abs(data))
    ax = sns.heatmap(data, square=True, cbar=False, cmap="vlag", **kwargs)
    ax.text(len(data[0])/2,len(data)/2, f"{sum:3.2f}", fontsize=16, ha="center", va="center")

def make_heatmaps(file, baseline):
    df = pd.read_csv(file)
    #print(df)

    # im = ["nomodel", "statecount", "maxentropy", "rnd", "grm"]
    # im_name = ["No IM", "State Count", "Max Entropy", "RND", "GRM"]
    # im = ["nomodel", "statecount", "maxentropy", "icm", "rnd", "grm"]
    # im_name = ["No IM", "State Count", "Max Entropy", "ICM", "RND", "GRM"]
    # im = ["nors+nomodel", "nors+statecount", "nors+maxentropy", "nors+icm", "grm+statecount", "grm+maxentropy", "grm+icm"]
    # im_name = ["No IM", "State Count", "Max Entropy", "ICM", "GRM+SC", "GRM+ME", "GRM+ICM"]
    # im = ["nors+nomodel", "nors+statecount", "nors+maxentropy", "nors+icm", "grm+statecount", "grm+maxentropy", "grm+icm", "adopes+statecount", "adopes+maxentropy", "adopes+icm"]
    # im_name = ["No IM", "State Count", "Max Entropy", "ICM", "GRM+SC", "GRM+ME", "GRM+ICM", "ADOPES+SC", "ADOPES+ME", "ADOPES+ICM"]
    # im = ["nors+nomodel", "nors+statecount", "nors+icm", "grm+statecount", "grm+icm", "adopes+statecount", "adopes+icm", "pies+statecount", "pies+icm"]
    # im_name = ["No IM", "State Count", "ICM", "GRM+SC", "GRM+ICM", "ADOPES+SC", "ADOPES+ICM", "PIES+SC", "PIES+ICM"]
    # im = ["nors+nomodel", "nors+statecount", "grm+statecount", "adopes+statecount",  "pies+statecount"]
    # im_name = ["No IM", "State Count", "GRM+SC", "ADOPES+SC", "PIES+SC"]
    im = ["nors+statecount", "nors+statecount+cir"]
    im_name = ["State Count", "SC+CIR"]
    df["im"] = df["im"].replace(im, im_name)
    im = im_name

    # sn1 = [3, 15, 305, 610, 915, 1221]
    # sn_name = ["0.25%", "1.25%", "25%", "50%", "75%", "100%"]
    # sn1 = [25, 49, 123, 245, 489]
    # sn1 = [50,100,250,500,1000]
    # sn1 = [250,500,1250,2500,5000]
    # sn_name = ["5%", "10%", "25%", "50%", "100%"]
    # sn_name = ["10%", "100%"]
    
    ref = re.search(r'positions-([A-Za-z0-9]+(\-\d+x\d+)?(\-(\w\d)+)*)(\-fixed([\d]+))?.csv', file)
    map = ref.group(1)
    seed  = ref.group(6)
    map_width, map_height, max_v, max_diff_v = map_dims[map]

    ims = im_name
    sn1 = snap_dict[map]
    snapshots = ["5%", "10%", "25%", "50%", "100%"]

    df["snapshot"] = df["snapshot"].replace(sn1, snapshots)
    # Filter by seed
    # df = df[df["seed"]==1]

    # Filter by snapshot
    # snapshots = ["5%", "100%"]

    big_map = []

    for im in ims:
        for snapshot in snapshots:
            # counts = np.zeros((mapsize, mapsize))
            sub_df = df.loc[ (df["im"]==im) & (df["snapshot"] == snapshot) ][["x","y"]]
            counts = sub_df.value_counts() / sub_df.shape[0]
            sum_df = np.array([ [0.0 if (x,y) not in counts.index else counts[x,y] for x in range(map_width)] for y in range(map_height) ])
            avg_rew = np.mean( df.loc[ (df["im"]==im) & (df["snapshot"] == snapshot) & (df["done"] == True) ][["reward"]] )
            total_cost = np.sum( df.loc[ (df["im"]==im) & (df["snapshot"] == snapshot) ][["cost"]], axis=0 ).item()
            big_map.append( (snapshot, im, sum_df, avg_rew, total_cost) )

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
@click.option('--file', type=str, help='CSV file with the positions of agents')
@click.option('--baseline', default="No IM", type=str, help='Name of the baseline method')

def main(
    file, baseline,
):
    make_heatmaps(file, baseline)
    

if __name__ == '__main__':
    main()
