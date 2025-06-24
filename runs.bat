SET map=DoorKey-8x8
SET rs=GRM
SET im=StateCount
SET irc=0.01
SET seeds=(6, 7, 8, 9, 10)

(for %%s in %seeds% do (
    python train.py --game_name=%map% --int_rew_source=%im% --int_rew_coef=%irc% --run_id=%%s --int_shape_source=%rs% --grm_delay=1 --model_recs=[50,100,250,500,1000]
))
