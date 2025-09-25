import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

convergence_return = {
    "Empty-16x16" : 0.976,
    "LavaCrossingS11N5" : 0.95,
}

def make_plots_tuning(file):
    # sns.set_theme(style="darkgrid")
    sns.set_theme(style="ticks", rc={'font.family':'serif', 'font.serif':'Times New Roman'})

    df = pd.read_csv(file)
    df = df.dropna(axis=0, subset=["iterations"])

    # atts = ["rollout/ep_rew_mean", "rollout/ll_unique_states", "rollout/ll_unique_positions", "rollout/ep_entropy:"]
    # atts_name = ["Episode Reward", "Position Coverage", "Observation Coverage", "Entropy"]
    atts = ["rollout/ep_rew_mean", "rollout/ll_unique_states", "rollout/ep_entropy", "rollout/ll_cost_count"]
    atts_name = ["Episode Reward", "Position Coverage", "Entropy", "Total Constraint Violations"]
    df = df.rename(columns=dict([(x, y) for x, y in zip(atts, atts_name)]))

    # maps = ["Empty-16x16", "DoorKey-16x16", "RedBlueDoors-8x8", "FourRooms"]
    # maps = ["Empty-16x16", "DoorKey-8x8", "RedBlueDoors-8x8", "FourRooms", "LavaCrossingS11N5", "MultiRoom-N4-S5"]
    maps = ["Empty-16x16", "LavaCrossingS11N5"]

    # Find convergence timestep
    min_steps = 25
    conv_df = []
    for map in maps:
        conv_ret = convergence_return[map]
        for seed in df.loc[df["map"]==map,"seed"].unique():
            for collision_cost in df.loc[df["map"]==map,"collision_cost"].unique():
                if seed == 3 and collision_cost in [1.0, 1.25, 1.5, 1.75]: continue 
                sub_df = df.loc[(df["map"] == map) & (df["seed"] == seed) & (df["collision_cost"] == collision_cost)]
                idx, combo = -1, 0
                for i in range(1, df.loc[(df["map"] == map), "iterations"].max()+1):
                    ret = sub_df.loc[(sub_df["iterations"] == i), "Episode Reward"].item()
                    if ret >= conv_ret:
                        combo += 1
                        if idx == -1: idx = i
                        if combo >= min_steps: break
                    else:
                        idx, combo = -1, 0
                if idx == -1: idx = df.loc[(df["map"] == map), "iterations"].max()
                conv_df.append([collision_cost, map, "Convergence", idx])
    conv_df = pd.DataFrame(conv_df, columns=["collision_cost", "map", "metric", "value"])
            


    # Remove all but last 50 interations
    for map in maps:
        df.loc[(df["map"] == map), "iterations"] -=  df.loc[(df["map"] == map), "iterations"].max()
    df = df[df["iterations"] > -50 ]

    df = df[["collision_cost", "map"] + atts_name].set_index(["collision_cost", "map"]).stack(future_stack=True).reset_index()
    df.columns = ["collision_cost", "map", "metric", "value"]
    df = pd.concat([df, conv_df])
    atts_name += ["Convergence"]

    # Make Position/State coverage relative to map
    for map in maps:
        for metric in ["Position Coverage"]:
            df.loc[(df["map"] == map) & (df["metric"] == metric), "value"] /= df[(df["map"] == map) & (df["metric"] == metric)]["value"].max()

    g = sns.FacetGrid(df, row="map", col="metric", row_order=maps, col_order=atts_name, sharex=False, sharey=False,
                        margin_titles=True, legend_out=True, despine=False, height=2.5, aspect=1.6)
    g.map_dataframe(sns.boxplot, x="collision_cost", y="value", notch=True, showcaps=False)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels("", "")
    # for (row_val, col_val), ax in g.axes_dict.items():
    #     # ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
    #     # ax.set_xticks([0, 200, 400, 600, 800, 1000])
    #     if col_val == "Entropy":
    #         # ax.set_ylim((0,2))
    #         # ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    #         ax.set_ylim((0, 0.4))
    #         ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    #     elif col_val == "Total Constraint Violations":
    #         pass
    #     else:
    #         # ax.set_ylim((0,1))
    #         # ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    #         ax.set_ylim((0.4,1))
    #         ax.set_yticks([0.4, 0.55, 0.7, 0.85, 1.0])
    #     ax.set_xticks(ticks_dict[row_val])
    # g.add_legend(ncol=len(sub_plot))
    g.tight_layout()
    plt.show()
    g.savefig(f"tuning.png", dpi=300)


@click.command()
@click.option('--file', type=str, default="logs/alldata.csv", help='CSV file with the training statistics of all models')

def main(
    file,
):
    make_plots_tuning(file)
 

if __name__ == '__main__':
    main()
