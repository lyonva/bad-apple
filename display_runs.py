import os
from os.path import join
import pandas as pd
import io

log_dir = "logs"
models_dir = "models"

maps = [ f.name for f in os.scandir(log_dir) if f.is_dir() ]

for map in maps:
    print(30*"-")
    print(map)
    logs = [ f.name for f in os.scandir(join(log_dir, map)) if f.is_dir() ]

    for log in logs:
        seeds = [ f.name for f in os.scandir(join(log_dir, map, log)) if f.is_dir() ]
        
        for seed in seeds:
            path_log = join(log_dir, map, log, seed)
            models_log = join(models_dir, map, log, seed)
            print(path_log)

            if os.path.exists(join(path_log, "rollout.csv")) and os.path.isfile(join(path_log, "rollout.csv")):
                df = pd.read_csv(join(path_log, "rollout.csv"), usecols=range(4))
                iter = int( df["iterations"].max() )
                
                

            else:
                iter = 0
                complete = False

    print(30*"-")


print(subfolders)
