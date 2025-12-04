import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from os.path import join


snap_dict = {
    "Empty-16x16" : [250,500,1250,2500,5000],
    "DoorKey-8x8" : [500,1000,2500,5000,10000],
    "RedBlueDoors-8x8" : [500,1000,2500,5000,10000],
    "FourRooms" : [1250,2500,6250,12500,25000],
    "LavaCrossingS11N5" : [500,1000,2500,5000,10000],
    "MultiRoom-N4-S5" : [500,1000,2500,5000,10000],
}


def display_test_reward(dir):
    sns.set_theme(style="ticks", rc={'font.family':'serif', 'font.serif':'Times New Roman'})

    maps = ["Empty-16x16", "DoorKey-8x8", "RedBlueDoors-8x8", "FourRooms", "LavaCrossingS11N5", "MultiRoom-N4-S5"]
    im_base = ["nors+nomodel", "nors+statecount", "grm+statecount", "adopes+statecount", "pies+statecount"]
    im_name = ["No IM", "State Count", "GRM", "ADOPS", "PIES"]
    snapshots = ["5%", "10%", "25%", "50%", "100%"]

    print(25*" " + "|".join( [ f"{s:11s}" for s in im_name ] ) )
    for map in maps:
        file = join(dir, f"rewards-{map}.csv")
        df = pd.read_csv(file)
        snaps = snap_dict[map]
        df["im"] = df["im"].replace(im_base, im_name)
        
        df["snapshot"] = df["snapshot"].replace(snaps, snapshots)

        row = []
        for im in im_name:
            sub_df = df[ (df["im"] == im) & (df["snapshot"] == "100%") ]
            row.append( np.mean(sub_df["reward"]) )
        print(f"{map:25s}" + "|".join( [ f"{r:11.4f}" for r in row ] ) )

    

@click.command()
@click.option('--dir', type=str, default=".", help='Directory with the data')

def main(
    dir,
):
    display_test_reward(dir)
    

if __name__ == '__main__':
    main()
