# Intrinsic Motivation Benchmarking for Reinforcement Learning
Clone of [__Discriminative-model-based Episodic Intrinsic Reward (DEIR)__](https://github.com/swan-utokyo/deir), adapted to Gymnasium 1.0 and MiniGrid 3.0. ProcGen was removed.

- Paper in [ArXiV]().
- Main, up to date [branch](https://github.com/lyonva).

## Usage
### Installation
Clone the repo, then:

```commandline
conda create -n bad-apple python=3.11
conda activate bad-apple
python -m pip install -r requirements.txt
```

Optionally, set up torch with cuda after installing the requirements:
- https://developer.nvidia.com/cuda-downloads
- https://pytorch.org/get-started/locally/

### Train PPO with no intrinsic rewards
```commandline
python train.py --env_source=minigrid --game_name=Empty-16x16 --int_rew_source=NoModel
```

### Train PPO+RND on MiniGrid
```commandline
python train.py --env_source=minigrid --game_name=DoorKey-16x16 --model_features_dim=64 --int_rew_source=RND
```

### Train and log results to wandb
run ```wandb init``` first

```commandline
python train.py --env_source=minigrid --game_name=DoorKey-16x16 --model_features_dim=64 --int_rew_source=RND --use_wandb=1 --run_id=1
```

The parameter ```run_id``` is the seed for the run.

### Make multiple runs on wandb
Depending on your system, use ```run.sh``` or ```run.bat``` to make your shell do multiple runs consecutively. You can adjust the variables within those scripts to alter the MiniGrid map, IM method and seeds.

### Replicating the Results of the Minding Motivation paper
Run the following line to generate one run:
```commandline
python train.py --game_name=[DoorKey-8x8|Empty-16x16|RedBlueDoors-8x8|FourRooms|DoorKey-16x16] --int_rew_source=[NoModel|StateCount|MaxEntropy|ICM] --int_rew_coef=IRC --run_id=SEED --int_shape_source=[NoRS|GRM] --grm_delay=1 --model_recs=[50,100,250,500,1000]
```

Set the values pertaining to the map, intrinsic reward source and whether or not to use GRM. Select the appropriate value for IRC from the table in the paper. Lastly, select a value for the seed. We ran seeds 1 through 10 for the paper.
The file `runs.bat` allows you to set these variables and run for multiple seed values.

Results are generated on the logs directory. Afterwards you can check which your runs are good enough (not interrupted midway, have all recordings) using `display_runs.py`:

```commandline
python display_runs.py
```

Successful runs will be transfered to analysis/logs, and will be appropriately renamed.

Afterwards you can generate the plots. For the metric plots (lines) run (note it will take a while):
```commandline
cd analysis
python utils.py
python make_plots_by_metric.py
```

You can also check which runs converge to a good enough policy:
```commandline
cd analysis
python display_runs_converge.py
```

For the behavior plots (heatmaps), we will first need to run the testing mode, which will load the trained models, run them in the environments and record their movements. On the root directory, run:
```commandline
python .\test.py --game_name=[DoorKey-8x8|Empty-16x16|RedBlueDoors-8x8|FourRooms|DoorKey-16x16] --models_dir=analysis\logs\MiniGrid-[DoorKey-8x8|Empty-16x16|RedBlueDoors-8x8|FourRooms|DoorKey-16x16]-v0 --baseline=nors+nomodel --snaps=[500,1000,2500,5000,10000] --fixed_seed=1
```

It will generate an appropriately named csv on the analysis directory. You can set `fixed_seed` to always test the models on the same map instance. Otherwise it will use the same seed the model was trained on. For the paper, we used the following seeds:
- DoorKey-8x8: 2
- Empty-16x16: 1
- RedBlueDoors-8x8: 1
- FourRooms: 7
- DoorKey-16x16: 1

Laslty, you can generate heatmaps with:
```commandline
cd analysis
python make_heatmaps.py --file=CSV_FILE_WITH_POSITIONS
```

On the unused difference heatmaps (red and blue over white), you can find the policy divergence values.
