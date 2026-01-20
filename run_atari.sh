#! /bin/bash
env_source="Atari"
map="MontezumaRevenge"
im="NoModel"
rs="NoRS"
steps=61440000
recs="7500,15000,22500,30000"
ext_rew_adjust=2
norm=0
dgrm=1
pies_decay=0
cost_as_ir=0
collision_cost=0
seeds=(0)

for seed in ${seeds[@]}; do
    python train.py --env_source="$env_source" --game_name="$map" --policy_cnn_type=1 --features_dim=448 --latents_dim=448 \
        --model_features_dim=448 --max_episode_steps=4500 --batch_size=128 --gamma=0.999 --ent_coef=0.001 \
        --learning_rate=0.0001 --model_learning_rate=0.00001 --policy_cnn_layers=2 --policy_mlp_layers=2 --model_mlp_layers=2 \
        --ext_rew_coef="$ext_rew_adjust" --int_rew_coef=1 --adv_eps=0.0000001 \
        --int_rew_source="$im" --int_rew_norm=$norm --run_id=$seed --int_shape_source="$rs" --grm_delay=$dgrm \
        --total_steps=$steps --model_recs=["$recs"] --pies_decay=$pies_decay --cost_as_ir=$cost_as_ir --collision_cost=$collision_cost
done

