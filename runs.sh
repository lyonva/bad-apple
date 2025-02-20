#! /bin/bash
map         =   DoorKey-16x16
im          =   NoModel
seeds       =   (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)

for seed in ${seeds[@]}; do
    python train.py --env_source=minigrid --game_name=$map --model_features_dim=64 --int_rew_source=$im --use_wandb=1 --run_id=$s
done
