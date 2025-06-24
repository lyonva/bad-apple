import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def make_plots(file):
    # sns.set_theme(style="darkgrid")
    sns.set_theme(style="ticks", rc={'font.family':'serif', 'font.serif':'Times New Roman'})

    df = pd.read_csv(file)
    # im = ["nomodel", "statecount", "grm"]
    # im_name = ["No IM", "State Count", "GRM"]
    # im = ["nomodel", "statecount", "maxentropy", "icm", "rnd", "grm"]
    # im_name = ["No IM", "State Count", "Max Entropy", "ICM", "RND", "GRM"]
    # im = ["nors+nomodel", "nors+statecount", "nors+maxentropy", "nors+icm", "grm+statecount", "grm+maxentropy", "grm+icm", "adopes+statecount", "adopes+maxentropy", "adopes+icm"]
    # im_name = ["No IM", "State Count", "Max Entropy", "ICM", "GRM+SC", "GRM+ME", "GRM+ICM", "ADOPES+SC", "ADOPES+ME", "ADOPES+ICM"]
    im = ["nors+nomodel", "nors+statecount", "nors+maxentropy", "nors+icm", "grm+statecount", "grm+maxentropy", "grm+icm"]
    im_name = ["No IM", "State Count", "Max Entropy", "ICM", "GRM+SC", "GRM+ME", "GRM+ICM"]
    df["im"] = df["im"].replace(im, im_name)
    im = im_name
    df = df.dropna(axis=0, subset=["iterations"])

    atts = ["rollout/ep_rew_mean", "rollout/ll_unique_positions", "rollout/ll_unique_states", "rollout/ep_entropy:"]
    atts_name = ["Episode Reward", "Position Coverage", "Observation Coverage", "Entropy"]
    df = df.rename(columns=dict([(x, y) for x, y in zip(atts, atts_name)]))
    df = df.astype({"Position Coverage" : "float32", "Observation Coverage" : "float32"})

    # maps = ["Empty-16x16", "DoorKey-16x16", "RedBlueDoors-8x8", "FourRooms"]
    # maps = ["Empty-16x16", "DoorKey-8x8", "RedBlueDoors-8x8", "FourRooms"]
    maps = ["DoorKey-8x8", "Empty-16x16", "FourRooms", "RedBlueDoors-8x8", "DoorKey-16x16"]
    maps_max_cov = [6*6, 14*14, 17*17, 6*14, 14*14]

    # Make Observation coverage relative to highest obtained per map
    for map in maps:
        df.loc[df["map"] == map, "Observation Coverage"] /= df[df["map"] == map]["Observation Coverage"].max()
    
    # Make Position coverage relative to highest theoretical
    for map, cov in zip(maps, maps_max_cov):
        df.loc[(df["map"] == map),  "Position Coverage"] /= cov

    for att in atts_name:

        att_df = df[["im", "iterations", "map", att]]

        # for map in maps_name:
        #     for metric in ["Episode Reward"]:
        #         print(map)
        #         print( df.loc[(df["map"] == map) & (df["metric"] == metric) & (df["iterations"] == 1221), ["im", "value"]].groupby("im").mean()  )

        # df = df[df["iterations"] <= 10] # Speed up for testing
        # df = df[df["iterations"] % 100 == 1 ] # Speed up for testing

        # sub_plots = [["No IM", "State Count", "Max Entropy", "ICM"],
        #              ["No IM", "State Count", "GRM+SC", "ADOPES+SC"],
        #              ["No IM", "Max Entropy", "GRM+ME", "ADOPES+ME"],
        #              ["No IM", "ICM", "GRM+ICM", "ADOPES+ICM"]
        # ]
        sub_plots = [["No IM", "State Count", "Max Entropy", "ICM"],
                    ["No IM", "State Count", "GRM+SC"],
                    ["No IM", "Max Entropy", "GRM+ME"],
                    ["No IM", "ICM", "GRM+ICM"]
        ]
        sub_group = ["Base IM", "State Count", "Max Entropy", "ICM"]

        pal = ["gray"] + sns.color_palette("Paired", 6)
        pal_order = ["No IM", "State Count", "GRM+SC", "Max Entropy", "GRM+ME", "ICM", "GRM+ICM"]
        # sub_colors = [[ "gray", pal[0], pal[2], pal[4] ],
        #               [ "gray", pal[0], pal[1] ],
        #               [ "gray", pal[2], pal[3] ],
        #               [ "gray", pal[4], pal[5] ],
        # ]

        ultra_frame = pd.DataFrame()
        for name, sub in zip(sub_group, sub_plots):
            sub_df = att_df[ np.isin(att_df["im"], sub) ].copy()
            sub_df["group"] = name
            ultra_frame = pd.concat([ultra_frame, sub_df])

        g = sns.FacetGrid(ultra_frame, row="group", col="map", col_order=maps, sharex=False, sharey=False,
                        margin_titles=True, legend_out=True, despine=False, height=2.5, aspect=1.8)
        g.map_dataframe(sns.lineplot, x="iterations", y=att, hue="im", hue_order=pal_order, palette=pal)
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        g.set_axis_labels("")
        for (row_val), ax in g.axes_dict.items():
            # ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
            ax.set_xticks([0, 200, 400, 600, 800, 1000])
            ax.set_ylim((0,1))
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            # ax.move_legend(ax, )
        g.add_legend(ncol=len(pal_order))
        g.tight_layout()
        plt.show()
        g.savefig(f"{att}.png", dpi=300)

@click.command()
@click.option('--file', type=str, default="logs/alldata.csv", help='CSV file with the training statistics of all models')

def main(
    file,
):
    make_plots(file)
    

if __name__ == '__main__':
    main()
