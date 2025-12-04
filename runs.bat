SET map=DoorKey-8x8
SET rs=GRM
SET im=StateCount
SET steps=20480000
SET recs=[500,1000,2500,5000,10000]
SET irc=0.01
SET seeds=(10)
SET adopes_coef=0.0002
SET pies_decay=5000

(for %%s in %seeds% do (
    python train.py --game_name=%map% --int_rew_source=%im% --int_rew_coef=%irc% --run_id=%%s --int_shape_source=%rs% --grm_delay=1 --total_steps=%steps% --model_recs=%recs% --adopes_coef_inc=%adopes_coef% --pies_decay=%pies_decay%
))
