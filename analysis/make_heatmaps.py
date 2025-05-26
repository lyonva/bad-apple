import click
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# map_dims = {
#     "Empty-16x16" : (16, 16, 0.05, 0.05),
#     "DoorKey-16x16" : (16, 16, 0.02, 0.02),
#     "RedBlueDoors-8x8" : (16, 8, 0.08, 0.08),
#     "FourRooms" : (19, 19, 0.01, 0.01),
# }

map_dims = {
    "Empty-16x16" : (16, 16, 0.04, 0.02),
    "DoorKey-8x8" : (8, 8, 0.1, 0.05),
    "RedBlueDoors-8x8" : (16, 8, 0.1, 0.05),
    "FourRooms" : (19, 19, 0.0075, 0.005),
}

def draw_log_heatmap(data, vmin=0, vmax=1, **kwargs):
    data = data.drop(["im", 'snapshot'], axis=1).iloc[0]["data"]
    # signs = np.logical_not(np.signbit(data)).astype(np.float32)*2 - 1
    # data = np.abs(data)
    # data = signs*np.log10(data - np.min(data) + 0.0001)
    data = np.log10(data - np.min(data) + 0.0001)
    sns.heatmap(data, square=True, cbar=False, vmin=np.log10(vmin+0.0001), vmax=np.log10(vmax), **kwargs)

def draw_heatmap(data, **kwargs):
    data = data.drop(["im", 'snapshot'], axis=1).iloc[0]["data"]
    sns.heatmap(data, square=True, cbar=False, **kwargs)


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

    map = file.split("\\")[-1].split(".")[0].split("-", 1)[1]
    map_width, map_height, max_v, max_diff_v = map_dims[map]

    ims = im_name
    snapshots = sn_name

    # Filter by seed
    # df = df[df["seed"]==1]

    big_map = []

    for im in ims:
        for snapshot in snapshots:
            # counts = np.zeros((mapsize, mapsize))
            sub_df = df.loc[ (df["im"]==im) & (df["snapshot"] == snapshot) ][["x","y"]]
            counts = sub_df.value_counts() / sub_df.shape[0]
            sum_df = np.array([ [0 if (x,y) not in counts.index else counts[x,y] for x in range(map_width)] for y in range(map_height) ])
            big_map.append( (snapshot, im, sum_df) )

    # Heatmap
    big_map = pd.DataFrame(big_map, columns=["snapshot", "im", "data"])
    g = sns.FacetGrid(big_map, row="im", col="snapshot", margin_titles=True)
    superheat=g.map_dataframe(draw_heatmap, annot=False, vmin=0, vmax=max_v)
    g.set_titles(col_template="Training: {col_name}", row_template="{row_name}")
    superheat.figure.savefig(f"heat-{map}.png")
    # plt.show()

    # Diff heatmap
    diff_map = []
    for snapshot in snapshots:
        base = big_map[(big_map["im"] == baseline) & (big_map["snapshot"] == snapshot)].drop(["im", 'snapshot'], axis=1).iloc[0]["data"]
        for im in ims:
            if im == baseline: continue
            tech = big_map[(big_map["im"] == im) & (big_map["snapshot"] == snapshot)].drop(["im", 'snapshot'], axis=1).iloc[0]["data"]
            tech -= base
            diff_map.append( (snapshot, im, tech) )
    
    diff_map = pd.DataFrame(diff_map, columns=["snapshot", "im", "data"])
    g = sns.FacetGrid(diff_map, row="im", col="snapshot", margin_titles=True)
    superheat=g.map_dataframe(draw_diff_heatmap, annot=False, vmin=-max_diff_v, vmax=max_diff_v)
    g.set_titles(col_template="Training: {col_name}", row_template="{row_name}")
    superheat.figure.savefig(f"diff-{map}.png")
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
