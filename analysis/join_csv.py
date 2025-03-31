import os
import ast
import pandas as pd

csvs = [f for f in os.listdir('.') if (f.endswith('.csv') and f != "alldata.csv")]

all_df = []

for csv in csvs:
    map = csv.split(".")[0].split("-")[1]
    df = pd.read_csv(csv)
    
    df["map"] = map
    
    all_df.append(df)

super_df = pd.concat(all_df)
print(super_df.groupby(["map", "im"]).count())

# print(super_df[(super_df["map"] == "DoorKey") & (super_df["im"] == "GRM")])

super_df.to_csv("alldata.csv")
