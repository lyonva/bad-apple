# About
Clone of [__Discriminative-model-based Episodic Intrinsic Reward (DEIR)__](https://github.com/swan-utokyo/deir), adapted to Gymnasium 1.0 and MiniGrid 3.0. ProcGen was removed.

# Usage
### Installation
Clone the repo, then:

```commandline
conda create -n bad-apple python=3.11
conda activate bad-apple
python3 -m pip install -r requirements.txt
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
