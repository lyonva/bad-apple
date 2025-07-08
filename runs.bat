SET map=DoorKey-16x16
SET rs=NoRS
SET im=NoModel
SET steps=5120000
SET recs=[125,250,625,1250,2500]
SET irc=0
SET seeds=(1,2,3)
SET adopes_coef=0.0008
SET pies_decay=1250

(for %%s in %seeds% do (
    python train.py --game_name=%map% --int_rew_source=%im% --int_rew_coef=%irc% --run_id=%%s --int_shape_source=%rs% --grm_delay=1 --total_steps=%steps% --model_recs=%recs% --adopes_coef_inc=%adopes_coef% --pies_decay=%pies_decay%
))
