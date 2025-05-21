#! /bin/bash
map="Empty-16x16"
im="NoModel"
irc=0.0005
rs="NoRS"
seeds=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

for seed in ${seeds[@]}; do
    python train.py --game_name="$map" --model_features_dim=64 --int_rew_source="$im" --int_rew_coef=$irc --run_id=$seed --int_shape_source="$rs" --grm_delay=1 --model_recs=[50,100,250,500,1000]
done
