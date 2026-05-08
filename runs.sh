#! /bin/bash
map="EmptyCenter-35x35"
rs="ADOPS"
im="StateCount"
irc=0.5
steps=20480000
recs="25,100,250,500,1000,10000"
norm=0
dgrm=5
pies_decay=5000
cost_as_ir=0
collision_cost=0
seeds=(1 2 3)

for seed in ${seeds[@]}; do
    python train.py --env_source="Minigrid" --game_name="$map" --int_rew_source="$im" --int_rew_coef=$irc --int_rew_norm=$norm --run_id=$seed --int_shape_source="$rs" --grm_delay=$dgrm \
        --total_steps=$steps --model_recs=["$recs"] --pies_decay=$pies_decay --cost_as_ir=$cost_as_ir --collision_cost=$collision_cost
done

