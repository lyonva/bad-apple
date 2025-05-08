import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_theme(style="darkgrid")
sns.set_theme(style="ticks", rc={'font.family':'serif', 'font.serif':'Times New Roman'})

df = pd.read_csv("logs/alldata.csv")
im = ["nomodel", "statecount", "maxentropy", "rnd", "grm"]
im_name = ["No IM", "State Count", "Max Entropy", "RND", "GRM"]
df["im"] = df["im"].replace(im, im_name)
im = im_name
df = df.dropna(axis=0, subset=["iterations"])

atts = ["rollout/ep_rew_mean", "rollout/ll_unique_states", "rollout/ll_unique_positions", "rollout/ep_entropy:"]
atts_name = ["Episode Reward", "State Coverage", "Position Coverage", "Entropy"]
df = df.rename(columns=dict([(x, y) for x, y in zip(atts, atts_name)]))

maps = ["Empty-16x16", "DoorKey-16x16", "RedBlueDoors-8x8", "FourRooms"]

df = df[["im", "iterations", "map"] + atts_name].set_index(["im", "iterations", "map"]).stack(future_stack=True).reset_index()
df.columns = ["im", "iterations", "map", "metric", "value"]

# Make Position/State coverage relative to map
for map in maps:
    for metric in ["State Coverage", "Position Coverage"]:
        df.loc[(df["map"] == map) & (df["metric"] == metric), "value"] /= df[(df["map"] == map) & (df["metric"] == metric)]["value"].max()

# for map in maps_name:
#     for metric in ["Episode Reward"]:
#         print(map)
#         print( df.loc[(df["map"] == map) & (df["metric"] == metric) & (df["iterations"] == 1221), ["im", "value"]].groupby("im").mean()  )

# df = df[df["iterations"] <= 10] # Speed up for testing
# df = df[df["iterations"] % 100 == 1 ] # Speed up for testing

g = sns.FacetGrid(df, row="map", col="metric", row_order=maps, col_order=atts_name, sharex=False, sharey=False,
                  margin_titles=True, legend_out=True, despine=False, height=2.5, aspect=1.6)
g.map_dataframe(sns.lineplot, x="iterations", y="value", hue="im", hue_order=im, palette='CMRmap')
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.set_axis_labels("", "")
for (row_val, col_val), ax in g.axes_dict.items():
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
    ax.set_xticks([0, 400, 800, 1200])
    if col_val == "Entropy":
        ax.set_ylim((0,2))
        ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    else:
        ax.set_ylim((0,1))
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
g.add_legend(ncol=len(im))
g.tight_layout()
plt.show()
g.savefig("all_plots.png", dpi=300)

