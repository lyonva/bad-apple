#! /bin/bash

map="Asteroids"
rs="NoRS"
im="NoModel"

seeds=(1 2 3 4 5)

dgrm=10
pies_decay=2500

steps=40960000
recs="2500,5000,7500,10000"

for seed in ${seeds[@]}; do
    python train.py --env_source="Atari" --game_name="$map" --num_processes=32 --policy_cnn_type=1 --model_cnn_type=1 --features_dim=512 --latents_dim=512 \
        --model_features_dim=512 --model_latents_dim=512 --max_episode_steps=10000 --batch_size=256 --n_steps=128 --n_epochs=4 --gamma=0.999 --ent_coef=0.01 \
        --learning_rate=0.0001 --model_learning_rate=0.00001 --policy_cnn_layers=2 --policy_mlp_layers=2 --model_mlp_layers=2 --policy_mlp_norm=BatchNorm --policy_cnn_norm=BatchNorm \
        --ext_rew_coef=2 --int_rew_coef=1 --adv_eps=0.0000001 --ent_coef=0.01 --clip_range_vf=-1 --max_grad_norm=0.5 --adv_norm=2 --model_cnn_norm=BatchNorm --model_mlp_norm=BatchNorm \
        --int_rew_source="$im" --int_rew_norm=1 --run_id=$seed --int_shape_source="$rs" --grm_delay=$dgrm \
        --total_steps=$steps --model_recs=["$recs"] --pies_decay=$pies_decay
done
