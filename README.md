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

### Train PPO+RND on MiniGrid
```commandline
python src/train.py \
  --int_rew_source=RND \
  --env_source=minigrid \
  --game_name=DoorKey-8x8 \
  --model_features_dim=64
```

### Train and log results to wandb
run ```wandb init``` first

```commandline
python src/train.py \
  --int_rew_source=RND \
  --env_source=minigrid \
  --game_name=DoorKey-8x8 \
  --model_features_dim=64 \
  --use_wandb=1
```

### Train DEIR on MiniGrid
Run the below command in the root directory of this repository to train a DEIR agent in the standard _DoorKey-8x8_ (MiniGrid) environment.
```commandline
python src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=DoorKey-8x8
```

### Train DEIR on MiniGrid with advanced settings
```commandline
python src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=DoorKey-8x8-ViewSize-3x3 \
  --can_see_walls=0 \
  --image_noise_scale=0.1
```

### Sample output in DoorKey-8x8
```
run:  0  iters: 1  frames: 8192  rew: 0.389687  rollout: 9.281 sec  train: 1.919 sec
run:  0  iters: 2  frames: 16384  rew: 0.024355  rollout: 9.426 sec  train: 1.803 sec
run:  0  iters: 3  frames: 24576  rew: 0.032737  rollout: 8.622 sec  train: 1.766 sec
run:  0  iters: 4  frames: 32768  rew: 0.036805  rollout: 8.309 sec  train: 1.776 sec
run:  0  iters: 5  frames: 40960  rew: 0.043546  rollout: 8.370 sec  train: 1.768 sec
run:  0  iters: 6  frames: 49152  rew: 0.068045  rollout: 8.337 sec  train: 1.772 sec
run:  0  iters: 7  frames: 57344  rew: 0.112299  rollout: 8.441 sec  train: 1.754 sec
run:  0  iters: 8  frames: 65536  rew: 0.188911  rollout: 8.328 sec  train: 1.732 sec
run:  0  iters: 9  frames: 73728  rew: 0.303772  rollout: 8.354 sec  train: 1.741 sec
run:  0  iters: 10  frames: 81920  rew: 0.519742  rollout: 8.239 sec  train: 1.749 sec
run:  0  iters: 11  frames: 90112  rew: 0.659334  rollout: 8.324 sec  train: 1.777 sec
run:  0  iters: 12  frames: 98304  rew: 0.784067  rollout: 8.869 sec  train: 1.833 sec
run:  0  iters: 13  frames: 106496  rew: 0.844819  rollout: 9.068 sec  train: 1.740 sec
run:  0  iters: 14  frames: 114688  rew: 0.892450  rollout: 8.077 sec  train: 1.745 sec
run:  0  iters: 15  frames: 122880  rew: 0.908270  rollout: 7.873 sec  train: 1.738 sec
```

### Example of Training Baselines
Please note that the default value of each option in `src/train.py` is optimized for DEIR. For now, when training other methods, please use the corresponding hyperparameter values specified in Table A1 (in our [arXiv preprint](https://arxiv.org/abs/2304.10770)). An example is `--int_rew_coef=3e-2` and `--rnd_err_norm=0` in the below command.
```commandline
python src/train.py \
  --int_rew_source=NovelD \
  --env_source=minigrid \
  --game_name=DoorKey-8x8 \
  --int_rew_coef=3e-2 \
  --rnd_err_norm=0
```
