#! /bin/bash
# map="SafeChoice"
# cost_limit=5.0
# map="SafeBogMaze"
# cost_limit=3.0
map="SafeWaterMaze-5x7"
cost_limit=2.0

# cost_objective="Lag"
# lagrange_learning_rate=0.00005
cost_objective="SB"
lagrange_learning_rate=0.00005
# cost_objective="SaBER"
# lagrange_learning_rate=0

# im="NoModel"
# rs="NoRS"
im="RND"
rs="ADOPES"
irc=0.5
steps=10240000
recs="250,500,1000,2500,5000"
norm=0
pies_decay=2500
saber_zeta_min_rollout=100
saber_zeta_max_rollout=250
seeds=(1 2 3 4 5 6 7 8 9 10)

for seed in ${seeds[@]}; do
    python train.py --env_source="Minigrid" --game_name="$map" --int_rew_source="$im" --int_rew_coef=$irc --int_rew_norm=$norm --run_id=$seed --int_shape_source="$rs" \
        --total_steps=$steps --model_recs=["$recs"] --pies_decay=$pies_decay --cost_critic=1 --cost_limit=$cost_limit --cost_objective=$cost_objective --lagrange_learning_rate=$lagrange_learning_rate \
        --saber_zeta_min_rollout=$saber_zeta_min_rollout --saber_zeta_max_rollout=$saber_zeta_max_rollout --model_learning_rate=0.0003 --rnd_err_norm=2
done

