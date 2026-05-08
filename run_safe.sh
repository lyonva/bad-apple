#! /bin/bash
map="SafeBogMaze"
cost_limit=2.0
im="NoModel"
rs="NoRS"
irc=0.5
steps=10240000
recs="5000"
norm=0
dgrm=10
pies_decay=2500
seeds=(1 2 3)

for seed in ${seeds[@]}; do
    python train.py --env_source="Minigrid" --game_name="$map" --int_rew_source="$im" --int_rew_coef=$irc --int_rew_norm=$norm --run_id=$seed --int_shape_source="$rs" --grm_delay=$dgrm \
        --total_steps=$steps --model_recs=["$recs"] --pies_decay=$pies_decay --cost_critic=1 --cost_limit=$cost_limit
done

