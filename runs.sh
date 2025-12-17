#! /bin/bash
env_source="Atari"
map="asteroids"
im="NoModel"
irc=0.0005
rs="NoRS"
steps=10240000
recs="250,500,1250,2500,5000"
irc=0.1
pies_decay=0
cost_as_ir=0
collision_cost=0
seeds=(1)

for seed in ${seeds[@]}; do
    python train.py --env_source="$env_source" --game_name="$map" --int_rew_source="$im" --int_rew_coef=$irc --run_id=$seed --int_shape_source="$rs" --grm_delay=1 --total_steps=$steps --model_recs=["$recs"] --pies_decay=$pies_decay --cost_as_ir=$cost_as_ir --collision_cost=$collision_cost
done

