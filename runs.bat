SET env_source=MiniGrid
SET map=Empty-16x16
SET rs=PBIM
SET im=StateCount
SET steps=2048000
SET recs=[1000]
SET irc=0.1
SET seeds=(1,2,3,4,5)
SET int_rew_norm=0
SET grm_delay=1
SET pies_decay=0
SET cost_as_ir=0
SET collision_cost=0

(for %%s in %seeds% do (
    python train.py --env_source=%env_source% --game_name=%map% --int_rew_source=%im% --int_rew_coef=%irc% --int_rew_norm=%int_rew_norm% --run_id=%%s --int_shape_source=%rs% --grm_delay=%grm_delay% --total_steps=%steps% --model_recs=%recs% --pies_decay=%pies_decay% --cost_as_ir=%cost_as_ir% --collision_cost=%collision_cost%
))
