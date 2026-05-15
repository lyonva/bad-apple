#! /bin/bash
# map="SafeChoice"
# cost_limit=5.0
# map="SafeBogMaze"
# cost_limit=3.0
# map="SafeWaterMaze-5x7"
# cost_limit=3
# map="SafeMoatMaze-8x8"
# cost_limit=5
map="SafeDetMoatMaze-8x8"
cost_limit=5
# map="SafeMockjocoGoal-9x9"
# cost_limit=0

# cost_objective="NoCO"
# cost_objective="Lag"
# cost_objective="SB"
cost_objective="SaBER"

im="RND"
rs="ADOPS"

# Safe Choice
# steps=409600
# recs="100,200"
# norm=0
# pies_decay=50
# lagrange_learning_rate=0.05
# saber_zeta_min_rollout=1
# saber_zeta_max_rollout=50

# SafeWaterMaze-5x7
# steps=2048000
# recs="19,37,75,125,250"
# irc=0.005
# norm=1
# pies_decay=125
# lagrange_learning_rate=0.02
# saber_zeta_min_rollout=1
# saber_zeta_max_rollout=75

# SafeDetMoatMaze-8x8
steps=32768000
recs="19,37,75,125,250,500,1000,1900,1950,2000,3000,3800,3900,4000"
irc=0.0025
norm=1
pies_decay=2000
lagrange_learning_rate=0.0005
saber_zeta_min_rollout=1
saber_zeta_max_rollout=1000

# SafeMockjocoGoal-9x9
# steps=40960000
# recs="19,37,75,125,250,500,1000,2000,3000,4000,5000"
# irc=0.0025
# norm=1
# pies_decay=2500
# lagrange_learning_rate=0.0005
# saber_zeta_min_rollout=1
# saber_zeta_max_rollout=1250

seeds=(1 2 3 4 5)
# seeds=(0)

for seed in ${seeds[@]};
do
    python train.py --env_source="Minigrid" --game_name="$map" --int_rew_source="$im" --int_rew_coef=$irc --int_rew_norm=$norm --run_id=$seed --int_shape_source="$rs" --total_steps=$steps --model_recs=["$recs"] --pies_decay=$pies_decay --cost_critic=1 --cost_limit=$cost_limit --cost_objective=$cost_objective --lagrange_learning_rate=$lagrange_learning_rate --saber_zeta_min_rollout=$saber_zeta_min_rollout --saber_zeta_max_rollout=$saber_zeta_max_rollout --learning_rate=0.0001
done

