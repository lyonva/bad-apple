
import click
import os
from os.path import join

snap_dict = {
    "MiniGrid-Empty-16x16-v0" : [250,500,1250,2500,5000],
    "MiniGrid-DoorKey-8x8-v0" : [500,1000,2500,5000,10000],
    "MiniGrid-RedBlueDoors-8x8-v0" : [500,1000,2500,5000,10000],
    "MiniGrid-FourRooms-v0" : [1250,2500,6250,12500,25000],
    "MiniGrid-LavaCrossingS11N5-v0" : [500,1000,2500,5000,10000],
    "MiniGrid-MultiRoom-N4-S5-v0" : [500,1000,2500,5000,10000],
}

def display_runs(dir):
    maps = [ f.name for f in os.scandir(dir) if f.is_dir() ]

    min_seed = 9999999
    max_seed = 0
    all_runs = dict([(map, {}) for map in maps])
    all_techs = set()
    for map in maps:
        map_dir = join(dir, map)
        snaps = snap_dict[map]
        runs = [ f.name for f in os.scandir(map_dir) if f.is_dir() ]
        
        for run in runs:
            tech, seed = run.split("-")
            seed = int(seed)
            if tech not in all_runs[map].keys(): all_runs[map][tech] = {}
            all_techs.add(tech)
            max_seed = max(max_seed, seed)
            min_seed = min(min_seed, seed)
            
            all_found = True
            for snap in snaps:
                if not(os.path.exists(join(dir, map, run, f'snapshot-{snap}.zip'))):
                    all_found = False
                    break
            all_runs[map][tech][seed] = all_found
    
    # Now display
    for map in maps:
        print(f"{map:35}" + "|".join( [ f"{s:2d}" for s in range(min_seed, max_seed+1) ] ))
        for tech in all_techs:
            
            print(f"{tech:35}" + "|".join( [ "❌" if (tech not in all_runs[map]) else ("❌" if (s not in all_runs[map][tech]) else ("✔️" if (all_runs[map][tech][s]) else "⚠️")) for s in range(min_seed,max_seed+1) ] ))
        print("")


@click.command()
# Testing params
@click.option('--dir', default="logs", type=str, help='Directory with the training logs')
def main(
    dir,
):
    display_runs(dir)
    

if __name__ == '__main__':
    main()
