import os
from os.path import join
import pandas as pd
import io
import shutil
from src.algo.ppo_trainer import PPOTrainer
from src.utils.enum_types import ModelType, ShapeType

def yes_or_no(question):
    while True:
        reply = str(input(question+' [Y/n]: ')).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False

def get_map_snaps(map_name):
    if map_name in snap_dict.keys():
        return snap_dict[map_name]
    return default_snaps

snap_dict = {
    "MiniGrid-Empty-16x16-v0" : [125,250,625,1250,2500],
    "MiniGrid-DoorKey-8x8-v0" : [500,1000,2500,5000,10000],
    "MiniGrid-RedBlueDoors-8x8-v0" : [1000,2000,5000,10000,20000],
    "MiniGrid-FourRooms-v0" : [1000,2000,5000,10000,20000],
}
default_snaps = [250,500,1250,2500,5000]


archive_dir = "analysis"
log_dir = "logs"
models_dir = "models"

maps = [ f.name for f in os.scandir(log_dir) if f.is_dir() ]

complete_runs = []
incomplete_runs = []
empty_dirs = []

for map in maps:
    # print(30*"-")
    # print(map)
    model_snap_tags = get_map_snaps(map)
    max_iter = model_snap_tags[-1]

    logs = [ f.name for f in os.scandir(join(log_dir, map)) if f.is_dir() ]

    for log in logs:
        seeds = [ f.name for f in os.scandir(join(log_dir, map, log)) if f.is_dir() ]
        if len(seeds) == 0: empty_dirs.append(join(log_dir, map, log))
        
        # Check empty model dirs
        if os.path.exists(join(models_dir, map, log)) and len( [ f.name for f in os.scandir(join(models_dir, map, log))] ) == 0: empty_dirs.append(join(models_dir, map, log))

        for seed in seeds:
            path_log = join(log_dir, map, log, seed)
            models_log = join(models_dir, map, log, seed)
            # print(path_log)
            code = 0

            if len([ f.name for f in os.scandir(path_log) ]) == 0: empty_dirs.append(path_log)
            if os.path.exists(models_log):
                if len([ f.name for f in os.scandir(models_log) ]) == 0: empty_dirs.append(models_log)

            if os.path.exists(join(path_log, "rollout.csv")) and os.path.isfile(join(path_log, "rollout.csv")):
                df = pd.read_csv(join(path_log, "rollout.csv"), usecols=range(4))
                iter = int( df["iterations"].max() )
                
                all_models = True
                a_model_path = None
                for mst in model_snap_tags:
                    if os.path.exists(join(models_log, f"snapshot-{mst}.zip")):
                        a_model_path = join(models_log, f"snapshot-{mst}.zip")
                    else:
                        all_models = False
                        code = 2
                
                if a_model_path is not None:
                # Try and get parameters
                    try:
                        model_main = PPOTrainer.load( a_model_path, env=None )
                        im = ModelType.get_str_name(model_main.int_rew_source)
                        rs = ShapeType.get_str_name(model_main.int_shape_source)
                        id = model_main.run_id
                        model_loads = True
                    except:
                        model_loads = False
                        code = 3

                    # print(im, id)
                if iter != max_iter:
                    code = 4

            else:
                iter = 0
                code = 1
            
            if code == 0:
                complete_runs.append( [ map, rs, im, id, log, path_log, models_log ] )
            else:
                incomplete_runs.append( [log, code, path_log, models_log] )


    # print(30*"-")


print(f"Complete runs: {len(complete_runs)}")
for run in complete_runs:
    print(f"{run[0]:30s}\t{run[1]}+{run[2]}-{run[3]:2d}\t{run[4]}")

print(f"Incomplete runs: {len(incomplete_runs)}")
for run in incomplete_runs:
    code = run[1]
    if code == 1: err = "No logs detected"
    if code == 2: err = "Snapshots missing"
    if code == 3: err = "Couldn't open model"
    if code == 4: err = "Not enough iterations"
    print(f"{run[0]}\t{err}")

# print(subfolders)

if len(incomplete_runs) > 0:
    if yes_or_no("Delete all incomplete runs?"):
        for run in incomplete_runs:
            path_log = run[2]
            models_log = run[3]
            if os.path.exists(path_log):
                shutil.rmtree(path_log)
                if os.path.exists(path_log):
                    os.rmdir(path_log)
            if os.path.exists(models_log):
                shutil.rmtree(models_log)
                if os.path.exists(models_log):
                    os.rmdir(models_log)

print(f"Empty directories: {len(empty_dirs)}")
if len(empty_dirs) > 0:
    if yes_or_no("Delete all empty dirs?"):
        for dir in empty_dirs:
            os.rmdir(dir)


if len(complete_runs) > 0:
    if yes_or_no(f"Migrate all complete runs to {archive_dir}/{log_dir}?"):
        overwrite = None
        this_session_targets = []
        archive_log_dir = os.path.join(archive_dir, log_dir)

        if not os.path.exists(archive_dir): os.mkdir(archive_dir)
        if not os.path.exists(archive_log_dir): os.mkdir(archive_log_dir)

        for run in complete_runs:
            map, rs, im, id, log, path_log, models_log = run

            if not os.path.exists( os.path.join(archive_log_dir, map) ): os.mkdir( os.path.join(archive_log_dir, map) )
            target_dir = os.path.join( archive_log_dir, map, f"{rs}+{im}-{id}" )

            # Dir/Overwrite check
            if os.path.exists( target_dir ):
                if target_dir in this_session_targets:
                    print(f"Batch already contains a log exported to {target_dir}, skipping exporting of {map}/{log}")
                    continue
                if overwrite == None:
                    overwrite = yes_or_no("Overwrite existing?")
                if overwrite:
                    shutil.rmtree(target_dir)
                    os.mkdir(target_dir)
                else:
                    print(f"Target directory already contains a log exported to {target_dir}, skipping exporting of {map}/{log}")
                    continue
            else:
                os.mkdir(target_dir)
            
            for file in os.listdir(path_log): # Logs
                shutil.move( os.path.join( path_log, file ), target_dir )
            for file in os.listdir(models_log): # Snapshots
                shutil.move( os.path.join( models_log, file ), target_dir )

            # Safe delete dirs
            if len([ f.name for f in os.scandir(path_log) ]) == 0: os.rmdir(path_log)
            if len([ f.name for f in os.scandir(models_log) ]) == 0: os.rmdir(models_log)

            this_session_targets.append(target_dir)

