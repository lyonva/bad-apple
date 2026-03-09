import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import import_config_module
from utils import import_config_module, get_map_snaps

def make_plots(file, config_file, convolve, group):
    df = pd.read_csv(file)
    config = import_config_module(config_file)

    df = df.dropna(axis=0, subset=["iterations"])
    # Atari: Remove iterations where no episodes end
    #df = df[df["rollout/ep_len_mean"] > 0]

    atts = config.metrics
    atts_name = config.metrics_name
    df = df.rename(columns=dict([(x, y) for x, y in zip(atts, atts_name)]))

    im = config.im
    im_name = config.im_name
    maps = config.maps
    maps_name = config.maps_name
    df = df.replace({"map" : dict([(x, y) for x, y in zip(maps, maps_name)]), "im" : dict([(x, y) for x, y in zip(im, im_name)])})
    
    ticks_dict = config.ticks_dict
    # ticks_dict = dict([(m, ticks_dict[ma]) for ma, m in zip(maps, maps_name) ])
    map_threshold = config.map_threshold
    
    im = im_name
    maps = maps_name

    seeds = config.seeds

    df = df[["im", "iterations", "map", "seed"] + atts_name].set_index(["im", "iterations", "map", "seed"]).stack(future_stack=True).reset_index()
    df.columns = ["Model", "iterations", "map", "seed", "metric", "value"]

    # print(df)

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
    # if convolve > 0:
    #     mask = np.ones((convolve,))/convolve
    #     new_df = []
    #     for map in maps:
    #         for seed in seeds:
    #             for ims in im:
    #                 for metric in atts_name:
    #                     data = df[(df["map"]==map) & (df["Model"]==ims)  & (df["seed"]==seed) & (df["metric"]==metric)]["value"].to_numpy()
    #                     iters_data = df[(df["map"]==map) & (df["Model"]==ims)  & (df["seed"]==seed) & (df["metric"]==metric)]["iterations"].to_numpy()
    #                     data = data[:ticks_dict[map][-1]]
    #                     conv_data = np.convolve(data, mask, mode='valid')
    #                     conv_data = np.concatenate((conv_data, np.repeat(conv_data[-1], convolve-1)))
    #                     conv_iters = np.convolve(iters_data, mask, mode='valid')
    #                     conv_iters = np.concatenate((conv_iters, np.repeat(conv_iters[-1], convolve-1)))
    #                     for i, v in zip(conv_iters, conv_data):
    #                         new_df.append([ims, i, map, seed, metric, v])
    #     df = pd.DataFrame(new_df, columns=["Model", "iterations", "map", "seed", "metric", "value"])

    
    if convolve > 0:
        mask = np.ones((convolve,))/convolve
        new_df = []
        for map in maps:
            max_iters = get_map_snaps(map, config.maps_snapshot, config.default_snaps)[-1]
            for seed in seeds:
                for ims in im:
                    for metric in atts_name:
                        data = df[(df["map"]==map) & (df["Model"]==ims)  & (df["seed"]==seed) & (df["metric"]==metric)]
                        # print(map, ims, seed, metric, data)
                        for i in range(1, max_iters+1):
                            avg = data[(data["iterations"] > (i - convolve/2)) & (data["iterations"] < (i + convolve/2))]["value"].mean()
                            new_df.append([ims, i, map, seed, metric, avg])


                        # conv_data = np.convolve(data, mask, mode='valid')
                        # conv_data = np.concatenate((conv_data, np.repeat(conv_data[-1], convolve-1)))
                        # conv_iters = np.convolve(iters_data, mask, mode='valid')
                        # conv_iters = np.concatenate((conv_iters, np.repeat(conv_iters[-1], convolve-1)))
                        # for i, v in zip(conv_iters, conv_data):
                        #     new_df.append([ims, i, map, seed, metric, v])
        df = pd.DataFrame(new_df, columns=["Model", "iterations", "map", "seed", "metric", "value"])
    
    
    # df = df[df["iterations"] <= 10] # Speed up for testing
    # df = df[df["iterations"] % 100 == 0 ] # Speed up for testing

    if group>0:
        sns.set_theme(style="ticks", rc={'font.family':'serif'})
        
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
                            margin_titles=True, legend_out=True, despine=False, height=2.5, aspect=1.6, errorbar="se")
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
                    # ax.set_ylim((0.4,1))
                    # ax.set_yticks([0.4, 0.55, 0.7, 0.85, 1.0])
                    pass
                ax.set_xticks(ticks_dict[row_val])
            g.add_legend(ncol=len(sub_plot))
            g.tight_layout()
            plt.show()
            g.savefig(f"all_plots-{i}.png", dpi=300)
    else:
        sns.set_theme(style="whitegrid", rc={'font.family':'serif', 'font.serif':'Times New Roman', 'figure.figsize':(9,6)})
        hues = ["black", "red", "lightgreen", "green", "lightblue", "blue", "pink", "purple"]
        lines = ["-", ":", "--", "-.", "--", "-.", "--", "-."]
        for i, map in enumerate(maps):
            for metric in atts_name:
                sub_df = df[(df["map"]==map) & (df["metric"]==metric)]
                ax = sns.lineplot(sub_df, x="iterations", y="value", hue="Model", style="Model", palette=hues, dashes=True, errorbar="se")
                ax.set_xticks(ticks_dict[map])
                # if metric == "Average Episode Reward":
                #     ax.axline(xy1=(0,map_threshold[ i ]), slope=0, color = 'gray', ls='--')
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
                    pass
                    # ax.set_ylim((0,1))
                    # ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
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
@click.option('--config', default='config', type=str, help='Config file')
@click.option('--convolve', type=int, default=100, help='Size of convolution window')
@click.option('--group', type=int, default=1, help='Whether to group the plots using FacetGrid')

def main(
    file, config, convolve, group,
):
    make_plots(file, config, convolve, group)
    

if __name__ == '__main__':
    main()
