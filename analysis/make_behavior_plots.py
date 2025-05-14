import click
import glob
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

map_dims = {
    "Empty-16x16" : (16, 16, 0.05, 0.05),
    # "DoorKey-16x16" : (16, 16, 0.02, 0.02),
    "DoorKey-8x8" : (8, 8, 0.05, 0.05),
    "RedBlueDoors-8x8" : (16, 8, 0.08, 0.08),
    "FourRooms" : (19, 19, 0.01, 0.01),
}

def draw_heatmap(data, **kwargs):
    data = data.drop(["im", 'snapshot'], axis=1).iloc[0]["data"]
    sns.heatmap(data, square=True, cbar=False, **kwargs)

def make_behavior_plots(dir, baseline):
    files = glob.glob(f"{dir}/positions-*.csv")

    big_map = []
    vectors = []
    maps = []
    
    for file in files:
        df = pd.read_csv(file)

        im = ["nomodel", "statecount", "maxentropy", "rnd", "grm"]
        im_name = ["No IM", "State Count", "Max Entropy", "RND", "GRM"]
        df["im"] = df["im"].replace(im, im_name)
        im = im_name

        sn1 = [3, 15, 305, 610, 915, 1221]
        sn_name = [0.0025, 0.0125, 0.25, 0.50, 0.75, 1.0]
        df["snapshot"] = df["snapshot"].replace(sn1, sn_name)

        map = file.split("\\")[-1].split(".")[0].split("-", 1)[1]
        map_width, map_height, max_v, max_diff_v = map_dims[map]
        maps.append(map)

        ims = im_name
        snapshots = sn_name
        seeds = df["seed"].unique()

        for im in ims:
            for snapshot in snapshots:
                for seed in seeds:
                    # counts = np.zeros((mapsize, mapsize))
                    sub_df = df.loc[ (df["im"]==im) & (df["snapshot"] == snapshot)  & (df["seed"] == seed) ][["x","y"]]
                    counts = sub_df.value_counts() / sub_df.shape[0]
                    sum_df = np.array([ [0 if (x,y) not in counts.index else counts[x,y] for x in range(map_width)] for y in range(map_height) ])
                    big_map.append( (map, snapshot, seed, im, sum_df) )

                    vec = [0, 0, 0, 0] # Up right down left
                    for env in range(df["n_env"].max()):
                        sub_df = df.loc[ (df["im"]==im) & (df["snapshot"] == snapshot)  & (df["seed"] == seed) & (df["n_env"] == env) ][["x","y"]]
                        px, py = None, None
                        for i, row in sub_df.iterrows():
                            x, y = row["x"], row["y"]
                            if px is not None and np.abs((x - px)) + np.abs((y - px)) == 1:
                                if y < py: # Up
                                    vec[0] += 1
                                if x < px: # Right
                                    vec[1] += 1
                                if y > py: # Down
                                    vec[2] += 1
                                if x > px: # Left
                                    vec[3] += 1
                            px, py = x, y
                    if np.linalg.norm(vec) != 0:
                        vec /= np.linalg.norm(vec)
                    else: vec = (0.5, 0.5, 0.5, 0.5)
                    vectors.append( (map, snapshot, im, seed, vec) )
        


    big_map = pd.DataFrame(big_map, columns=["map", "snapshot", "seed", "im", "data"])
    vectors = pd.DataFrame(vectors, columns=["map", "snapshot", "im", "seed", "vector"])

    # Gather metrics
    metrics_map = []
    for map in maps:
        for im in ims:
            for snapshot in snapshots:
                for seed in seeds:
                    counts = big_map[(big_map["map"] == map) & (big_map["im"] == im) & (big_map["snapshot"] == snapshot) & (big_map["seed"] == seed)].drop(["map", "im", 'snapshot', 'seed'], axis=1).iloc[0]["data"]
                    baseline_counts = big_map[(big_map["map"] == map) & (big_map["im"] == baseline) & (big_map["snapshot"] == snapshot) & (big_map["seed"] == seed)].drop(["map", "im", 'snapshot', 'seed'], axis=1).iloc[0]["data"]
                    vector = vectors[(vectors["map"] == map) & (vectors["im"] == im) & (vectors["snapshot"] == snapshot) & (vectors["seed"] == seed)].drop(["map", "im", 'snapshot', 'seed'], axis=1).iloc[0]["vector"]
                    baseline_vector = vectors[(vectors["map"] == map) & (vectors["im"] == baseline) & (vectors["snapshot"] == snapshot) & (vectors["seed"] == seed)].drop(["map", "im", 'snapshot', 'seed'], axis=1).iloc[0]["vector"]
                    exp_entropy = 0
                    diff = 0
                    divergence = 0
                    div_count = 0
                    for i in range(len(counts)):
                        for j in range(len(counts[i])):
                            if counts[i][j] > 0:
                                exp_entropy -= counts[i][j] * np.log(counts[i][j])
                            diff += np.abs(counts[i][j] - baseline_counts[i][j])
                            if baseline_counts[i][j] > 0:
                                divergence += counts[i][j] / baseline_counts[i][j]
                                div_count += 1
                    if div_count > 0:
                        divergence /= div_count
                    exp_entropy /= np.log( len(counts) * len(counts[i]) )
                    cosine_similarity = np.dot(vector, baseline_vector) # Already unit vectors
                    metrics_map.append( (map, snapshot, im, seed, "Exploration Entropy", exp_entropy) )
                    metrics_map.append( (map, snapshot, im, seed, "State Difference", diff) )
                    metrics_map.append( (map, snapshot, im, seed, "Policy Similarity", cosine_similarity) )
                    # metrics_map.append( (map, snapshot, im, seed, "State Divergence", divergence) )
    
    metrics_map = pd.DataFrame(metrics_map, columns=["map", "snapshot", "im", "seed", "metric", "value"])

    g = sns.FacetGrid(metrics_map, row="map", col="metric", sharex=False, sharey=False, row_order=map_dims.keys(),
                  margin_titles=True, legend_out=True, despine=False, height=2.5, aspect=1.6)
    g.map_dataframe(sns.lineplot, x="snapshot", y="value", hue="im", hue_order=ims, palette='CMRmap')
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels("", "")
    for (row_val, col_val), ax in g.axes_dict.items():
        ax.set_xticks([0.0025, 0.25, 0.50, 0.75, 1.0])
        if col_val == "State Difference":
            ax.set_ylim((0,2))
            ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
        else:
            ax.set_ylim((0,1))
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    g.add_legend(ncol=len(ims))
    g.tight_layout()
    plt.show()
    g.savefig("all_behaviors.png", dpi=300)


@click.command()
# Testing params
@click.option('--dir', default=".", type=str, help='Directory with the CSV files')
@click.option('--baseline', default="No IM", type=str, help='Name of the baseline method')

def main(
    dir, baseline,
):
    make_behavior_plots(dir, baseline)
    

if __name__ == '__main__':
    main()
