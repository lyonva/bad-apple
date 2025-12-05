SET map=Empty-16x16
SET rs=NoRS
SET im=NoModel
SET steps=10240000
SET recs=[250,500,1250,2500,5000]
SET irc=0.1
SET seeds=(1,2,3)
SET pies_decay=0
SET cost_as_ir=1
SET collision_cost=1.50

(for %%s in %seeds% do (
    python train.py --game_name=%map% --int_rew_source=%im% --int_rew_coef=%irc% --run_id=%%s --int_shape_source=%rs% --grm_delay=1 --total_steps=%steps% --model_recs=%recs% --pies_decay=%pies_decay% --cost_as_ir=%cost_as_ir% --collision_cost=%collision_cost%
))
