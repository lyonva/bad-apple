#! /bin/bash
env_source="MiniGrid"
map="Empty-16x16"
im="NoModel"
irc=0
rs="NoRS"
steps=2048000
recs="1000"
norm=0
dgrm=1
pies_decay=0
cost_as_ir=0
collision_cost=0
seeds=(1,2,3,4,5)

for seed in ${seeds[@]}; do
    python train.py --env_source="$env_source" --game_name="$map" --int_rew_source="$im" --int_rew_coef=$irc --int_rew_norm=$norm --run_id=$seed --int_shape_source="$rs" --grm_delay=$dgrm --total_steps=$steps --model_recs=["$recs"] --pies_decay=$pies_decay --cost_as_ir=$cost_as_ir --collision_cost=$collision_cost
done

