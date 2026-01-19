# Intrinsic Motivation Benchmarking for Reinforcement Learning
Clone of [__Discriminative-model-based Episodic Intrinsic Reward (DEIR)__](https://github.com/swan-utokyo/deir), adapted to Gymnasium 1.0 and MiniGrid 3.0. ProcGen was removed.
Code base for running, benchmarking, and analyzing the results of RL agents with intrinsic motivation, safety, and risk.

## Installation
Clone the repo, then install dependencies:

### Venv
```commandline
python -m venv venv/bad-apple
source ./venv/bad-apple/bin/activate
python -m pip install -r requirements.txt
```

### Conda
```commandline
conda create -n bad-apple python=3.11
conda activate bad-apple
python -m pip install -r requirements.txt
```

### GPU Acceleration
Optionally, set up torch with cuda after installing the requirements:
- https://developer.nvidia.com/cuda-downloads
- https://pytorch.org/get-started/locally/

You can check if cuda is functioning:
```commandline
python
import torch
torch.cuda.is_available()
```
The above call should return True.

### Running the code
Once everything is installed, you can check if the code is functioning by running a simple trial
```commandline
python train.py --env_source=minigrid --game_name=Empty-8x8 --int_rew_source=NoModel --total_steps=4096
```

This should create a recording in logs/MiniGrid-Empty-8x8-v0


## Usage
### Train PPO with no intrinsic rewards on MiniGrid Empty-16x16
```commandline
python train.py --env_source=minigrid --game_name=Empty-16x16 --int_rew_source=NoModel
```

### Train PPO+RND on MiniGrid on DoorKey
```commandline
python train.py --env_source=minigrid --game_name=DoorKey-8x8 --model_features_dim=64 --int_rew_source=RND
```

The parameter ```run_id``` is the seed for the run.

### Batch files
For running multiple experiments in succession, as well as customizing parameters, see the provided .bat/.sh files.


## Results and Analysis
Results are generated in the logs directory. Afterwards, you can check whether your runs are good enough (not interrupted midway, have all recordings) using `display_runs.py`:

```commandline
python display_runs.py
```

Successful runs will be transferred to analysis/logs, and will be appropriately renamed.

Display run checks that the recorded trials have enough iterations and model checkpoints. These can be customized in analysis/config.py

(Under construction)

Afterwards, you can generate the plots. For the metric plots (lines) run (note it will take a while):
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

