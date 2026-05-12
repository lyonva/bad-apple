#! /bin/bash
# map="SafeChoice"
# cost_limit=5.0
# map="SafeBogMaze"
# cost_limit=3.0
map="SafeWaterMaze-5x7"
cost_limit=2.5

# cost_objective="NoCO"
# cost_objective="Lag"
# cost_objective="SB"
cost_objective="SaBER"

im="RND"
rs="ADOPS"
irc=0.0025


# Safe Choice
# steps=409600
# recs="100,200"
# norm=0
# pies_decay=50
# lagrange_learning_rate=0.05
# saber_zeta_min_rollout=1
# saber_zeta_max_rollout=50

# SafeWaterMaze-5x7
steps=4096000
recs="19,37,75,125,250,500"
norm=1
pies_decay=350
lagrange_learning_rate=0.01
saber_zeta_min_rollout=1
saber_zeta_max_rollout=150

seeds=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

for seed in ${seeds[@]};
do
    python train.py --env_source="Minigrid" --game_name="$map" --int_rew_source="$im" --int_rew_coef=$irc --int_rew_norm=$norm --run_id=$seed --int_shape_source="$rs" \
        --total_steps=$steps --model_recs=["$recs"] --pies_decay=$pies_decay --cost_critic=1 --cost_limit=$cost_limit --cost_objective=$cost_objective --lagrange_learning_rate=$lagrange_learning_rate \
        --saber_zeta_min_rollout=$saber_zeta_min_rollout --saber_zeta_max_rollout=$saber_zeta_max_rollout --clip_range_vf=0.5 --learning_rate=0.0001
done

