SET map=Empty-16x16
SET rs=NoRS
SET im=NoModel
SET irc=0.001
SET seeds=(1, 2, 3, 4, 5)

(for %%s in %seeds% do (
    python train.py --game_name=%map% --int_rew_source=%im% --int_rew_coef=%irc% --run_id=%%s --int_shape_source=%rs% --grm_delay=1 --model_recs=[50,100,250,500,1000]
))
