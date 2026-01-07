import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ticks_dict = {
#     "Empty-16x16" : [0,1000,2000,3000,4000,5000],
#     "DoorKey-8x8" : [0,2000,4000,6000,8000,10000],
#     "RedBlueDoors-8x8" : [0,2000,4000,6000,8000,10000],
#     "FourRooms" : [0,5000,10000,15000,20000,25000],
#     "LavaCrossingS11N5" : [0,2000,4000,6000,8000,10000],
#     "MultiRoom-N4-S5" : [0,2000,4000,6000,8000,10000],
# }

ticks_dict = {
    "Empty-16x16" : [0,200,400,600,800,1000],
    "DoorKey-8x8" : [0,200,400,600,800,1000],
    "RedBlueDoors-8x8" : [0,400,800,1200,1600,2000],
    "FourRooms" : [0,400,800,1200,1600,2000],
    "LavaCrossingS11N5" : [0,400,800,1200,1600,2000],
    "MultiRoom-N4-S5" : [0,400,800,1200,1600,2000],
}

def make_plots(file, convolve, group):
    df = pd.read_csv(file)
    # im = ["nomodel", "statecount", "grm"]
    # im_name = ["No IM", "State Count", "GRM"]
    # im = ["nomodel", "statecount", "maxentropy", "icm", "rnd", "grm"]
    # im_name = ["No IM", "State Count", "Max Entropy", "ICM", "RND", "GRM"]
    # im = ["nors+nomodel", "nors+statecount", "nors+maxentropy", "nors+icm", "grm+statecount", "grm+maxentropy", "grm+icm", "adopes+statecount", "adopes+maxentropy", "adopes+icm"]
    # im_name = ["No IM", "State Count", "Max Entropy", "ICM", "GRM+SC", "GRM+ME", "GRM+ICM", "ADOPES+SC", "ADOPES+ME", "ADOPES+ICM"]
    # im = ["nors+nomodel", "nors+statecount", "grm+statecount", "adopes+statecount", "pies+statecount"]
    # im_name = ["No IM", "State Count", "GRM+SC", "ADOPES+SC", "PIES+SC"]
    # im = ["nors+statecount", "nors+statecount+cir", "adopes+statecount+cir", "adopes+statecount+cirs"]
    # im_name = ["State Count", "State Count CIR", "SC+ADOPES+CIR", "SC+ADOPES+CIRS"]
    # im = ["NoRS+NoModel+cir+collision_cost0.25", "NoRS+NoModel+cir+collision_cost0.75", "NoRS+NoModel+cir+collision_cost2.0", "NoRS+NoModel+cir+collision_cost2.5"]
    # im_name = ["Cost 0.25", "Cost 0.75", "Cost 2.0", "Cost 2.5"]
    im = ["NoRS+NoModel+grm_delay1+int_rew_norm0", "NoRS+StateCount+grm_delay1+int_rew_norm0", "PBIM+StateCount+grm_delay1+int_rew_norm0", "PBIM+StateCount+grm_delay1+int_rew_norm1",
          "GRM+StateCount+grm_delay1+int_rew_norm0", "GRM+StateCount+grm_delay1+int_rew_norm1", "GRM+StateCount+grm_delay10+int_rew_norm0", "GRM+StateCount+grm_delay10+int_rew_norm1"]
    im_name = ["No IM", "IM", "PBIM no norm", "PBIM", "GRM D=1 no norm", "GRM D=1", "GRM D=10 no norm", "GRM D=10"]
    df["im"] = df["im"].replace(im, im_name)
    im = im_name
    df = df.dropna(axis=0, subset=["iterations"])

    # atts = ["rollout/ep_rew_mean", "rollout/ll_unique_states", "rollout/ll_unique_positions", "rollout/ep_entropy:"]
    # atts_name = ["Episode Reward", "Position Coverage", "Observation Coverage", "Entropy"]
    # atts = ["rollout/ep_rew_mean", "rollout/ll_unique_states", "rollout/ep_entropy", "rollout/ll_cost_count"]
    # atts_name = ["Episode Reward", "Position Coverage", "Entropy", "Total Constraint Violations"]
    atts = ["rollout/ep_rew_mean", "rollout/ep_len_mean"]
    atts_name = ["Average Episode Reward", "Average Episode Length"]
    df = df.rename(columns=dict([(x, y) for x, y in zip(atts, atts_name)]))

    # maps = ["Empty-16x16", "DoorKey-16x16", "RedBlueDoors-8x8", "FourRooms"]
    # maps = ["Empty-16x16", "DoorKey-8x8", "RedBlueDoors-8x8", "FourRooms", "LavaCrossingS11N5", "MultiRoom-N4-S5"]
    maps = ["Empty-16x16", "DoorKey-8x8", "LavaCrossingS11N5", "MultiRoom-N4-S5"]

    df = df[["im", "iterations", "map", "seed"] + atts_name].set_index(["im", "iterations", "map", "seed"]).stack(future_stack=True).reset_index()
    df.columns = ["Model", "iterations", "map", "seed", "metric", "value"]

    # Make Position/State coverage relative to map
    # for map in maps:
    #     # for metric in ["Position Coverage", "Observation Coverage"]:
    #     for metric in ["Position Coverage"]:
    #         df.loc[(df["map"] == map) & (df["metric"] == metric), "value"] /= df[(df["map"] == map) & (df["metric"] == metric)]["value"].max()

    # for map in maps_name:
    #     for metric in ["Episode Reward"]:
    #         print(map)
    #         print( df.loc[(df["map"] == map) & (df["metric"] == metric) & (df["iterations"] == 1221), ["Model", "value"]].groupby("Model").mean()  )

    # Convolve the plots
    if convolve > 0:
        mask = np.ones((convolve,))/convolve
        new_df = []
        for map in maps:
            for seed in range(1,df["seed"].max()+1):
                for ims in im:
                    for metric in atts_name:
                        data = df[(df["map"]==map) & (df["Model"]==ims)  & (df["seed"]==seed) & (df["metric"]==metric)]["value"].to_numpy()
                        data = data[:ticks_dict[map][-1]]
                        conv_data = np.convolve(data, mask, mode='valid')
                        conv_data = np.concatenate((conv_data, np.repeat(conv_data[-1], convolve-1)))
                        for i, v in enumerate(conv_data):
                            new_df.append([ims, i+1, map, seed, metric, v])
        df = pd.DataFrame(new_df, columns=["Model", "iterations", "map", "seed", "metric", "value"])
    
    
    # df = df[df["iterations"] <= 10] # Speed up for testing
    # df = df[df["iterations"] % 100 == 0 ] # Speed up for testing

    if group>0:
        sns.set_theme(style="ticks", rc={'font.family':'serif', 'font.serif':'Times New Roman'})
        
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
        # sub_plots = [["No IM", "State Count", "GRM+SC", "ADOPES+SC", "PIES+SC"]]
        sub_plots = [im_name.copy()]

        for i, sub_plot in enumerate(sub_plots):
            sub_df = df[ np.isin(df["Model"], sub_plot) ]
            g = sns.FacetGrid(sub_df, row="map", col="metric", row_order=maps, col_order=atts_name, sharex=False, sharey=False,
                            margin_titles=True, legend_out=True, despine=False, height=2.5, aspect=1.6)
            g.map_dataframe(sns.lineplot, x="iterations", y="value", hue="Model", hue_order=sub_plot, palette='CMRmap')
            g.set_titles(col_template="{col_name}", row_template="{row_name}")
            g.set_axis_labels("", "")
            for (row_val, col_val), ax in g.axes_dict.items():
                # ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
                # ax.set_xticks([0, 200, 400, 600, 800, 1000])
                if col_val == "Entropy":
                    # ax.set_ylim((0,2))
                    # ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
                    ax.set_ylim((0, 0.4))
                    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
                elif col_val == "Total Constraint Violations":
                    pass
                elif col_val == "Average Episode Length":
                    pass
                else:
                    # ax.set_ylim((0,1))
                    # ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                    ax.set_ylim((0.4,1))
                    ax.set_yticks([0.4, 0.55, 0.7, 0.85, 1.0])
                ax.set_xticks(ticks_dict[row_val])
            g.add_legend(ncol=len(sub_plot))
            g.tight_layout()
            plt.show()
            g.savefig(f"all_plots-{i}.png", dpi=300)
    else:
        sns.set_theme(style="whitegrid", rc={'font.family':'serif', 'font.serif':'Times New Roman', 'figure.figsize':(9,6)})
        hues = ["black", "red", "lightgreen", "green", "lightblue", "blue", "pink", "purple"]
        lines = ["-", ":", "--", "-.", "--", "-.", "--", "-."]
        for map in maps:
            for metric in atts_name:
                sub_df = df[(df["map"]==map) & (df["metric"]==metric)]
                ax = sns.lineplot(sub_df, x="iterations", y="value", hue="Model", style="Model", palette=hues, dashes=True)
                ax.set_xticks(ticks_dict[map])
                if metric == "Entropy":
                    # ax.set_ylim((0,2))
                    # ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
                    ax.set_ylim((0, 0.4))
                    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
                elif metric == "Total Constraint Violations":
                    pass
                elif metric == "Average Episode Length":
                    pass
                else:
                    ax.set_ylim((0,1))
                    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                    # ax.set_ylim((0.4,1))
                    # ax.set_yticks([0.4, 0.55, 0.7, 0.85, 1.0])
                ax.set_xlabel("Iterations")
                ax.set_ylabel(metric)
                fig = ax.get_figure()
                fig.tight_layout()
                fig.savefig(f"plot-{map}-{metric}.png", dpi=300)
                fig.clear()

@click.command()
@click.option('--file', type=str, default="logs/alldata.csv", help='CSV file with the training statistics of all models')
@click.option('--convolve', type=int, default=100, help='Size of convolution window')
@click.option('--group', type=int, default=1, help='Whether to group the plots using FacetGrid')

def main(
    file, convolve, group,
):
    make_plots(file, convolve, group)
    

if __name__ == '__main__':
    main()
