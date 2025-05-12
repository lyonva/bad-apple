SET map=Empty-16x16
SET im=NoModel
SET irc=0.0005
SET seeds=(1, 2, 3, 4, 5)

(for %%s in %seeds% do (
    python train.py --game_name=%map% --model_features_dim=64 --int_rew_source=%im% --int_rew_coef=%irc% --run_id=%%s --grm_delay=1 --model_recs=[25,49,123,245,489]
))
