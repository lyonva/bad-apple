import os
from os.path import join
import pandas as pd
import io
import shutil
from src.algo.ppo_trainer import PPOTrainer
from src.utils.enum_types import ModelType, ShapeType
import pickle

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
    "MiniGrid-Empty-16x16-v0" : [250,500,1250,2500,5000],
    "MiniGrid-DoorKey-8x8-v0" : [500,1000,2500,5000,10000],
    "MiniGrid-RedBlueDoors-8x8-v0" : [500,1000,2500,5000,10000],
    "MiniGrid-FourRooms-v0" : [1250,2500,6250,12500,25000],
    "MiniGrid-LavaCrossingS11N5-v0" : [500,1000,2500,5000,10000],
    "MiniGrid-MultiRoom-N4-S5-v0" : [500,1000,2500,5000,10000],
}
default_snaps = [250,500,1250,2500,5000]

# Parameters we want to distinguish
parameters = ['collision_cost']

archive_dir = "analysis"
log_dir = "logs"

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

        for seed in seeds:
            path_log = join(log_dir, map, log, seed)
            # print(path_log)
            code = 0

            if len([ f.name for f in os.scandir(path_log) ]) == 0: empty_dirs.append(path_log)

            if os.path.exists(join(path_log, "rollout.csv")) and os.path.isfile(join(path_log, "rollout.csv")):
                df = pd.read_csv(join(path_log, "rollout.csv"), usecols=range(4))
                iter = int( df["iterations"].max() )
                
                all_models = True
                # Check all model snapshot files are present
                for mst in model_snap_tags:
                    if not(os.path.exists(join(path_log, f"snapshot-{mst}.zip"))):
                        all_models = False
                        code = 2
                        break
                
                if all_models:
                # Try and get parameters
                    if os.path.exists(join(path_log, "params.pkl")) and os.path.isfile(join(path_log, "params.pkl")):
                        # try:
                            with open(join(path_log, "params.pkl"), 'rb') as f:
                                loaded_params = dict(pickle.load(f))
                            im = loaded_params["int_rew_source"]
                            rs = loaded_params["int_shape_source"]
                            ci = loaded_params["cost_as_ir"]
                            pp = {}
                            for param in parameters:
                                pp[param] = loaded_params[param]
                            id = loaded_params["run_id"]
                            params_load = True
                        # except:
                        #     params_load = False
                        #     code = 3
                    else:
                        params_load = False
                        code = 3

                    # print(im, id)
                if iter != max_iter:
                    code = 4

            else:
                iter = 0
                code = 1
            
            if code == 0:
                complete_runs.append( [ map, rs, im, ci, pp, id, log, path_log ] )
            else:
                incomplete_runs.append( [log, code, path_log] )


    # print(30*"-")


print(f"Complete runs: {len(complete_runs)}")
for run in complete_runs:
    ciro = '+cir' if run[3] == 1 else '+cirs' if run[3] == 2 else ''
    print(f"{run[0]:30s}\t{run[1]}+{run[2]}{ciro}-{run[5]:2d}\t{run[4]}\t{run[6]}")

print(f"Incomplete runs: {len(incomplete_runs)}")
for run in incomplete_runs:
    code = run[1]
    if code == 1: err = "No logs detected"
    if code == 2: err = "Snapshots missing"
    if code == 3: err = "Parameters not found"
    if code == 4: err = "Not enough iterations"
    print(f"{run[0]}\t{err}")

# print(subfolders)

if len(incomplete_runs) > 0:
    if yes_or_no("Delete all incomplete runs?"):
        for run in incomplete_runs:
            path_log = run[2]
            if os.path.exists(path_log):
                shutil.rmtree(path_log)
                if os.path.exists(path_log):
                    os.rmdir(path_log)

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
            map, rs, im, ci, pp, id, log, path_log = run

            if not os.path.exists( os.path.join(archive_log_dir, map) ): os.mkdir( os.path.join(archive_log_dir, map) )
            ciro = '+cir' if ci == 1 else '+cirs' if ci == 2 else ''
            pps = '' if len(parameters) > 0 else '+' + '+'.join([f'{k}{v}' for k, v in pp.items()])
            target_dir = os.path.join( archive_log_dir, map, f"{rs}+{im}{ciro}{pps}-{id}" )

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

            # Safe delete dirs
            if len([ f.name for f in os.scandir(path_log) ]) == 0: os.rmdir(path_log)

            this_session_targets.append(target_dir)

