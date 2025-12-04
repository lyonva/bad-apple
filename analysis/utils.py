import pandas as pd
import os
import shutil

def load_all_training_data(path):
    dirs = [a for a in os.listdir(path) if os.path.isdir(os.path.join(path, a))]
    maps = ["-".join(x.split("-")[1:-1]) for x in dirs]
    df = pd.DataFrame()
    
    for dir, map in zip(dirs, maps):
        for subdir in [a for a in os.listdir(os.path.join(path, dir)) if os.path.isdir(os.path.join(path, dir, a))]:
            im, seed = subdir.split("-")
            seed = int(seed)

            with open(os.path.join(path, dir, subdir, "rollout.csv")) as file:
                data = []
                for line in file.readlines():
                    line = line.split(",")
                    if line[-1] == "\n":
                        line = line[:-1]
                    data.append(line)
            # print(data)
            # print([len(x) for x in data])
            subdf = pd.DataFrame(data[1:], columns=data[0])
            subdf["map"] = map
            subdf["im"] = im
            subdf["seed"] = seed
            df = pd.concat((df, subdf))
    
    return df

def fix_the_info_len_bug_training_data(path):
    dirs = [a for a in os.listdir(path) if os.path.isdir(os.path.join(path, a))]
    maps = ["-".join(x.split("-")[1:-1]) for x in dirs]
    
    for dir, map in zip(dirs, maps):
        for subdir in [a for a in os.listdir(os.path.join(path, dir)) if os.path.isdir(os.path.join(path, dir, a))]:
            im, seed = subdir.split("-")
            seed = int(seed)

            with open(os.path.join(path, dir, subdir, "rollout.csv")) as file:
                data = []
                save = False
                check = None
                for line in file.readlines():
                    line = line.split(",")
                    if line[-1] == "\n":
                        line = line[:-1]
                    if check == None: check = len(line)
                    else:
                        if check != len(line):
                            # Mismatch found, insert data here
                            print(map, subdir)
                            idx = data[0].index("rollout/ep_info_rew_mean") + 1
                            data[0].insert(idx, "rollout/ep_info_len_mean")
                            for d in data[1:]:
                                d.insert(idx, 0)
                            save = True
                            check = len(line)
                    data.append(line)
            if save:
                subdf = pd.DataFrame(data[1:], columns=data[0])
                subdf.to_csv(os.path.join(path, dir, subdir, "rollout.csv"))
    

if __name__ == "__main__":
    dir = "logs"
    # fix_the_info_len_bug_training_data(dir)
    df = load_all_training_data(dir)
    df.to_csv(os.path.join(dir, "alldata.csv")) 
    print(df)