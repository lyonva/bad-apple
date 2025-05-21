
import click
import os
from os.path import join


def display_runs(dir):
    maps = [ f.name for f in os.scandir(dir) if f.is_dir() ]

    min_seed = 9999999
    max_seed = 0
    all_runs = dict([(map, {}) for map in maps])
    all_techs = set()
    for map in maps:
        map_dir = join(dir, map)
        runs = [ f.name for f in os.scandir(map_dir) if f.is_dir() ]
        
        for run in runs:
            tech, seed = run.split("-")
            seed = int(seed)
            if tech not in all_runs[map].keys(): all_runs[map][tech] = []
            all_techs.add(tech)
            all_runs[map][tech].append(seed)
            max_seed = max(max_seed, seed)
            min_seed = min(min_seed, seed)
    
    # Now display
    for map in maps:
        print(f"{map:35}" + "|".join( [ f"{s:2d}" for s in range(min_seed, max_seed+1) ] ))
        for tech in all_techs:
            print(f"{tech:35}" + "|".join( [ "✔️" if s in all_runs[map][tech] else "❌" for s in range(min_seed,max_seed+1) ] ))
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
