SET map=LavaCrossingS11N5
SET rs=ADOPES
SET im=StateCount
SET steps=20480000
SET recs=[500,1000,2500,5000,10000]
SET irc=0.1
SET seeds=(1,2,3)
SET pies_decay=2500
SET cost_as_ir=2

(for %%s in %seeds% do (
    python train.py --game_name=%map% --int_rew_source=%im% --int_rew_coef=%irc% --run_id=%%s --int_shape_source=%rs% --grm_delay=1 --total_steps=%steps% --model_recs=%recs% --pies_decay=%pies_decay% --cost_as_ir=%cost_as_ir%
))
