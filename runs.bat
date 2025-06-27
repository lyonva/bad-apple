SET map=RedBlueDoors-8x8
SET rs=GRM
SET im=StateCount
SET irc=0.05
SET seeds=(1,2,3)

(for %%s in %seeds% do (
    python train.py --game_name=%map% --int_rew_source=%im% --int_rew_coef=%irc% --run_id=%%s --int_shape_source=%rs% --grm_delay=1 --model_recs=[250,500,1250,2500,5000]
))
