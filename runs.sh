#! /bin/bash
map="Empty-16x16"
im="NoModel"
irc=0.0005
rs="NoRS"
steps=10240000
recs="250,500,1250,2500,5000"
irc=0.1
pies_decay=0
cost_as_ir=1
collision_cost=1
seeds=(1 2 3 4 5 6 7 8 9 10)

for seed in ${seeds[@]}; do
    python train.py --game_name="$map" --int_rew_source="$im" --int_rew_coef=$irc --run_id=$seed --int_shape_source="$rs" --grm_delay=1 --total_steps=$steps --model_recs=["$recs"] --pies_decay=$pies_decay --cost_as_ir=$cost_as_ir --collision_cost=$collision_cost
done

