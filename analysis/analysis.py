import pandas as pd
import numpy as np

df = pd.read_csv("alldata.csv")
im = ["NoModel", "StateCount", "MaxEntropy", "RND", "GRM"]
df = df.dropna(axis=0, subset=["iterations"])

atts = ["rollout/ep_rew_mean", "rollout/ll_unique_states", "rollout/ll_unique_positions", "rollout/ep_entropy:", "rollout/int_rew_mean", "rollout/ll_rew_count"]
atts_name = ["Episode Reward", "State Coverage", "Position Coverage", "Entropy", "Intrinsic Reward", "Reward Count"]
df = df.rename(columns=dict([(x, y) for x, y in zip(atts, atts_name)]))

maps = ["DoorKey", "RedBlue", "FourRooms"]
maps_name = ["DoorKey-16x16", "RedBlueDoors-8x8", "FourRooms"]
df["map"] = df["map"].replace(maps, maps_name)

df = df[["im", "iterations", "map", "seed"] + atts_name].set_index(["im", "iterations", "map", "seed"]).stack(future_stack=True).reset_index()
df.columns = ["im", "iterations", "map", "seed", "metric", "value"]


# Make Position/State coverage relative to map
for map in maps_name:
    for metric in ["State Coverage", "Position Coverage"]:
        df.loc[(df["map"] == map) & (df["metric"] == metric), "value"] /= df[(df["map"] == map) & (df["metric"] == metric)]["value"].max()

# Calculate table of timestep in which we find real rewards
counts = {}
for map in maps_name:
    counts[map] = {}
    for imm in im:
        counts[map][imm] = {"first":[], "second":[], "third":[], "tenth":[], "hundreth":[], "thousandth":[], "ten thousandth":[]}
        for seed in df["seed"].unique():
            count = 0
            sub_df = df[ (df["map"] == map) & (df["im"] == imm) & (df["seed"] == seed) & (df["metric"] == "Reward Count") ][["iterations", "value"]].sort_values("iterations")
            for i, row in sub_df.iterrows():
                iter = row["iterations"]
                num = row["value"]
                if count < 1 and num >= 1:
                    counts[map][imm]["first"].append(iter)
                    count = 1
                if count < 2 and num >= 2:
                    counts[map][imm]["second"].append(iter)
                    count = 2
                if count < 3 and num >= 3:
                    counts[map][imm]["third"].append(iter)
                    count = 3
                if count < 10 and num >= 10:
                    counts[map][imm]["tenth"].append(iter)
                    count = 10
                if count < 100 and num >= 100:
                    counts[map][imm]["hundreth"].append(iter)
                    count = 100
                if count < 1000 and num >= 1000:
                    counts[map][imm]["thousandth"].append(iter)
                    count = 1000
                if count < 10000 and num >= 10000:
                    counts[map][imm]["ten thousandth"].append(iter)
                    count = 10000
                    break
            
            if count < 10000: counts[map][imm]["ten thousandth"].append(iter+1)
            if count < 1000: counts[map][imm]["thousandth"].append(iter+1)
            if count < 100: counts[map][imm]["hundreth"].append(iter+1)
            if count < 10: counts[map][imm]["tenth"].append(iter+1)
            if count < 3: counts[map][imm]["third"].append(iter+1)
            if count < 2: counts[map][imm]["second"].append(iter+1)
            if count < 1: counts[map][imm]["first"].append(iter+1)

for map in maps_name:
    for imm in im:
        print(f"{map} - {imm}")
        for key,vals in counts[map][imm].items():
            mean, std = np.mean(vals), np.std(vals)
            print(f"{key}: ${mean:0.1f}\pm{std:0.1f}$")
        print()
