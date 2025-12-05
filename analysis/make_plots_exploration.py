import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

xlims_dict = {
    "Empty-16x16" : [200, 200, 50],
    "DoorKey-8x8" : [500, 500, 50],
    "RedBlueDoors-8x8" : [500, 500, 50],
    "FourRooms" : [500, 500, 50],
    "LavaCrossingS11N5" : [500, 500, 50],
    "MultiRoom-N4-S5" : [500, 500, 50],
}

def make_plots_exploration(file):
    # sns.set_theme(style="darkgrid")
    sns.set_theme(style="ticks", rc={'font.family':'serif', 'font.serif':'Times New Roman'})

    df = pd.read_csv(file)
    # im = ["nomodel", "statecount", "grm"]
    # im_name = ["No IM", "State Count", "GRM"]
    # im = ["nomodel", "statecount", "maxentropy", "icm", "rnd", "grm"]
    # im_name = ["No IM", "State Count", "Max Entropy", "ICM", "RND", "GRM"]
    # im = ["nors+nomodel", "nors+statecount", "nors+maxentropy", "nors+icm", "grm+statecount", "grm+maxentropy", "grm+icm", "adopes+statecount", "adopes+maxentropy", "adopes+icm"]
    # im_name = ["No IM", "State Count", "Max Entropy", "ICM", "GRM+SC", "GRM+ME", "GRM+ICM", "ADOPES+SC", "ADOPES+ME", "ADOPES+ICM"]
    im = ["nors+nomodel", "nors+statecount", "grm+statecount", "adopes+statecount", "pies+statecount"]
    im_name = ["No IM", "State Count", "GRM", "ADOPS", "PIES"]
    df["im"] = df["im"].replace(im, im_name)
    im = im_name
    df = df.dropna(axis=0, subset=["iterations"])

    atts = ["rollout/ll_unique_states", "rollout/ll_unique_positions", "rollout/ll_rew_count"]
    atts_name = ["Position Coverage", "Observation Coverage", "Extrinsic Reward Instances"]
    df = df.rename(columns=dict([(x, y) for x, y in zip(atts, atts_name)]))

    # maps = ["Empty-16x16", "DoorKey-16x16", "RedBlueDoors-8x8", "FourRooms"]
    maps = ["Empty-16x16", "DoorKey-8x8", "RedBlueDoors-8x8", "FourRooms", "LavaCrossingS11N5", "MultiRoom-N4-S5"]

    df = df[["im", "iterations", "map"] + atts_name].set_index(["im", "iterations", "map"]).stack(future_stack=True).reset_index()
    df.columns = ["im", "iterations", "map", "metric", "value"]

    # Make Position/State coverage relative to map
    for map in maps:
        for metric in ["Position Coverage", "Observation Coverage"]:
            df.loc[(df["map"] == map) & (df["metric"] == metric), "value"] /= df[(df["map"] == map) & (df["metric"] == metric)]["value"].max()

    # for map in maps_name:
    #     for metric in ["Episode Reward"]:
    #         print(map)
    #         print( df.loc[(df["map"] == map) & (df["metric"] == metric) & (df["iterations"] == 1221), ["im", "value"]].groupby("im").mean()  )

    # Limit to certain iterations
    subdfs = []
    for map in maps:
        for metric, limit in zip(atts_name, xlims_dict[map]):
            subdfs.append( df[ (df["map"] == map) & (df["metric"] == metric) & (df["iterations"] <= limit) ] )
    df = pd.concat(subdfs)

    # df = df[df["iterations"] <= 10] # Speed up for testing
    # df = df[df["iterations"] % 50 == 1 ] # Speed up for testing

    # sub_plots = [["No IM", "State Count", "Max Entropy", "ICM"],
    #              ["No IM", "State Count", "GRM+SC", "ADOPES+SC"],
    #              ["No IM", "Max Entropy", "GRM+ME", "ADOPES+ME"],
    #              ["No IM", "ICM", "GRM+ICM", "ADOPES+ICM"]
    # ]
    # sub_plots = [["No IM", "State Count", "Max Entropy", "ICM"],
    #              ["No IM", "State Count", "GRM+SC"],
    #              ["No IM", "Max Entropy", "GRM+ME"],
    #              ["No IM", "ICM", "GRM+ICM"]
    # ]
    sub_plots = [["No IM", "State Count", "GRM", "ADOPS", "PIES"]]

    for i, sub_plot in enumerate(sub_plots):
        sub_df = df[ np.isin(df["im"], sub_plot) ]
        g = sns.FacetGrid(sub_df, row="map", col="metric", row_order=maps, col_order=atts_name, sharex=False, sharey=False,
                        margin_titles=True, legend_out=True, despine=False, height=2.5, aspect=1.6)
        g.map_dataframe(sns.lineplot, x="iterations", y="value", hue="im", hue_order=sub_plot, palette='CMRmap')
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        g.set_axis_labels("", "")
        g.add_legend(ncol=len(sub_plot))
        g.tight_layout()
        plt.show()
        g.savefig(f"exploration_plots-{i}.png", dpi=300)

@click.command()
@click.option('--file', type=str, default="logs/alldata.csv", help='CSV file with the training statistics of all models')

def main(
    file,
):
    make_plots_exploration(file)
    

if __name__ == '__main__':
    main()
