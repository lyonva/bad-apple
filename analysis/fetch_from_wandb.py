import pandas as pd
import wandb
import os
import shutil

proj = "DoorKey-16x16"
model_names = [ f"snapshot-{s}" for s in [3,15,305,610,915,1221] ]

api = wandb.Api()
entity, project = "lyonva-nc-state-university", f"MiniGrid-{proj}-v0"
runs = api.runs(entity + "/" + project)
names = []

row_list, counts = [], {}

for run in runs:
    name = run.name
    im, seed = name.split("-")
    seed = int(seed)
    if run.state == "finished" and name not in names and seed <= 20 and run.summary["iterations"] == 1221:
        names.append(name)

        if im not in counts:
            counts[im] = []
        counts[im].append(seed)

        for row in run.scan_history(page_size=3000):
            row["im"], row["seed"] = im, seed
            row_list.append(row)
        
        model_dir = os.path.join("models", proj, f"{im}", f"{seed}")
        os.makedirs(model_dir, exist_ok=True)

        all_art_names = [f"{name}_{mname}" for mname in model_names]
        for art in run.logged_artifacts(per_page=1000):
            s_name = art.name.split(":")[0]
            if s_name in all_art_names:
                dl_path = art.download()
                for f in os.listdir(dl_path):
                    shutil.move(os.path.join(dl_path, f), model_dir )

        

runs_df = pd.DataFrame.from_records( row_list )

runs_df.to_csv(f"{entity}--{proj}.csv")

print(runs_df.groupby(["im", "seed"]).count())
for key, val in counts.items():
    print(f"{key} ({len(val)}: {val})")

