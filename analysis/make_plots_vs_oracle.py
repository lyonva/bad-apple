import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from os.path import join

max_steps = {
    "Empty-16x16" : 1024,
    "DoorKey-8x8" : 640,
    "RedBlueDoors-8x8" : 1280,
    "FourRooms" : 100,
    "LavaCrossingS11N5" : 484,
    "MultiRoom-N4-S5" : 120,
}

snap_dict = {
    "Empty-16x16" : [250,500,1250,2500,5000],
    "DoorKey-8x8" : [500,1000,2500,5000,10000],
    "RedBlueDoors-8x8" : [500,1000,2500,5000,10000],
    "FourRooms" : [1250,2500,6250,12500,25000],
    "LavaCrossingS11N5" : [500,1000,2500,5000,10000],
    "MultiRoom-N4-S5" : [500,1000,2500,5000,10000],
}

threshold_dist = {
    "Empty-16x16" : - 0.9 * (2/1024),
    "DoorKey-8x8" : - 0.9 * (5/1024),
    "RedBlueDoors-8x8" : - 0.9 * (5/1024),
    "FourRooms" : - 0.9 * (35/1024),
    "LavaCrossingS11N5" :  - 0.9 * (10/1024),
    "MultiRoom-N4-S5" : - 0.9 * (10/1024),
}

def make_plots_vs_oracle(dir):
    sns.set_theme(style="ticks", rc={'font.family':'serif', 'font.serif':'Times New Roman'})

    maps = ["Empty-16x16", "DoorKey-8x8", "RedBlueDoors-8x8", "FourRooms", "LavaCrossingS11N5", "MultiRoom-N4-S5"]
    im_base = ["nors+nomodel", "nors+statecount", "grm+statecount", "adopes+statecount", "pies+statecount"]
    im_name = ["No IM", "State Count", "GRM", "ADOPS", "PIES"]
    snapshots = ["5%", "10%", "25%", "50%", "100%"]
    # snapshots = [0.05, 0.1, 0.25, 0.5, 1.0]
    big_map = []

    for map in maps:
        file = join(dir, f"rewards-{map}.csv")
        df = pd.read_csv(file)
        snaps = snap_dict[map]
        df["im"] = df["im"].replace(im_base, im_name)

        

        df["snapshot"] = df["snapshot"].replace(snaps, snapshots)

        for im in im_name:
            for snap in snapshots:
                sub_df = df[ (df["im"] == im) & (df["snapshot"] == snap) ]
                step_diff = sub_df["steps"] - sub_df["steps_oracle"]
                error = sub_df["reward"] - sub_df["reward_oracle"]
                adjusted_reward = 1 - 0.9 * ( step_diff / max_steps[map] )
                big_map.append([map, im, snap, "Step Difference", np.mean(step_diff)])
                big_map.append([map, im, snap, "Error", np.mean(error)])
                big_map.append([map, im, snap, "Adjusted Reward", np.mean(adjusted_reward)])
    
    big_df = pd.DataFrame(big_map, columns=["map", "im", "snapshot", "metric", "value"])

    big_df = big_df[ big_df["metric"] == "Error" ]
    
    g = sns.FacetGrid(big_df, row="map", col="metric", row_order=maps, sharex=False, sharey=False,
                    margin_titles=True, legend_out=True, despine=False, height=2.5, aspect=1.6)
    g.map_dataframe(sns.lineplot, x="snapshot", y="value", hue="im", hue_order=im_name, palette='CMRmap')
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels("", "")
    for (row_val, col_val), ax in g.axes_dict.items():
        ax.axhline( threshold_dist[row_val], linestyle = 'dashed', color="black", linewidth=0.75)
        # ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
        ax.set_xticks(snapshots)
        ax.set_xticklabels(["5%", "10%", "25%", "50%", "100%"])
        # if col_val == "Entropy":
        #     # ax.set_ylim((0,2))
        #     # ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
        #     ax.set_ylim((0, 0.4))
        #     ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
        # else:
        #     # ax.set_ylim((0,1))
        #     # ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        #     ax.set_ylim((0.4,1))
        #     ax.set_yticks([0.4, 0.55, 0.7, 0.85, 1.0])
    g.add_legend(ncol=len(im_name))
    g.tight_layout()
    # plt.show()
    g.savefig(f"vs_oracle.png", dpi=300)

    for map in maps:
        print(f"{map:24s}|" + "|".join([f'{(x):8s}' for x in snapshots])+"|")
        for im in im_name:
            d = [ big_df.loc[(big_df["map"] == map) & (big_df["im"] == im) & (big_df["snapshot"] == snap)]["value"].item() for snap in snapshots ]
            xd = [x >= threshold_dist[map] for x in d]
            print(f"{im:24s}|" + "|".join([f'{x:7.5f}' + ("*" if xdd else " ") for x, xdd in zip(d,xd)])+"|")
        print()

@click.command()
@click.option('--dir', type=str, default=".", help='Directory with the data')

def main(
    dir,
):
    make_plots_vs_oracle(dir)
    

if __name__ == '__main__':
    main()
